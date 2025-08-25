import os
import argparse
from datetime import datetime
import time

import numpy as np
import torch
from torch.optim.lr_scheduler import OneCycleLR, MultiplicativeLR, CosineAnnealingLR

import methods
import data_load
import data_tools
import utils
from models.lenet import LeNet
from models.resnet import ResNet18, ResNet34
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
# models with dropout for USDNL algorithm:
from models.lenet import LeNetDO
from models.resnet import ResNet18DO, ResNet34DO
import warnings
warnings.filterwarnings("ignore", message=".*intraop threads.*")
import importlib, amortized.inference_net
importlib.reload(amortized.inference_net)
from amortized.inference_net import InferenceNet
from torch.cuda.amp import GradScaler
import matplotlib.pyplot as plt
from math import sqrt
torch.backends.cudnn.benchmark = True
from torchvision.transforms import InterpolationMode











parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', type=str, help = 'dir to save result txt files', default='results/')
parser.add_argument('--root_dir', type=str, help = 'dir that stores datasets', default='data/')
parser.add_argument('--dataset', type=str, help='[mnist, cifar10, cifar100, food101]', default='mnist')
parser.add_argument('--method', type=str, help='[regular, rlvi, arlvi_zscore, coteaching, jocor, cdr, usdnl, bare]', default='rlvi')

###---for A-RLVI ---###
parser.add_argument('--tau', type=float, default=1.0,
                    help='temperature for detached target q_i(τ)')
parser.add_argument('--beta', type=float, default=1.0,
                    help='Weight for the KL divergence regularization term (zscore A-RLVI)')

parser.add_argument('--download', dest='download', action='store_true',
                    help='Download dataset if not present')
parser.add_argument('--no-download', dest='download', action='store_false',
                    help='Do not download dataset')
parser.set_defaults(download=True)

parser.add_argument('--warmup_epochs', type=int, default=2,
                    help='Number of warm-up epochs where π̄ is fixed (default: 2)')
parser.add_argument('--ema_alpha', type=float, help='momentum in ema average', default=0.95)
parser.add_argument('--beta_entropy_reg', type=float, help='coefficient for entropy regularization strength', default=0.05)
parser.add_argument('--lr_init', type=float, default=1e-3,
                    help='Initial learning rate for model (used by SGD)')
parser.add_argument('--lr_inference', type=float, default=5e-5,
                    help='Learning rate for the inference network (Adam)')
parser.add_argument('--split_percentage', type=float, help="fraction of noisy train kept for training (rest goes to validation)", default=0.95)
parser.add_argument('--eval_val_every',  type=int, default=1,
                    help='run val set every N epochs (−1 = never)')
parser.add_argument('--eval_test_every', type=int, default=1,
                    help='run test set every N epochs (−1 = never)')
parser.add_argument('--early_stop', action='store_true',
                    help='Enable early stopping on validation accuracy')
parser.add_argument('--early_stop_patience', type=int, default=8,
                    help='Stop if val acc does not improve for N epochs')
parser.add_argument('--wd_backbone',  type=float, default=1e-4,
                    help='Weight decay for backbone (decay params only)')
parser.add_argument('--wd_head',      type=float, default=5e-4,
                    help='Weight decay for classifier head (decay params only)')
parser.add_argument('--wd_inference', type=float, default=1e-4,
                    help='Weight decay for inference net')


###---###
parser.add_argument('--n_epoch', type=int, help='number of epochs for training', default=80)
parser.add_argument('--batch_size', type=int, help='batch size for training', default=64)

parser.add_argument('--wd', type=float, help='Weight decay for optimizer', default=1e-4)
parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0)
parser.add_argument('--noise_type', type=str, help='[pairflip, symmetric, asymmetric, instance]', default='pairflip')
parser.add_argument('--momentum', type=float, help='momentum', default=0.9)
parser.add_argument('--print_freq', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=4, help='number of subprocesses for data loading')
parser.add_argument('--seed', type=int, default=1)
# For alternative methods
parser.add_argument('--forget_rate', type=float, help='forget rate', default=None)
parser.add_argument('--num_gradual', type=int, default=10, help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
parser.add_argument('--exponent', type=float, default=1, help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
parser.add_argument('--debug', action='store_true',
                    help='Print debugging information during training')




args = parser.parse_args()
# --- histories for plotting ---
hist_ce, hist_kl = [], []
hist_grad_bb, hist_grad_cls, hist_grad_inf = [], [], []
hist_pi_min, hist_pi_max, hist_pi_mean = [], [], []

# LR traces (flat over all batches)
hist_lr_bb, hist_lr_cls = [], []
# Test accuracy history (sampled only when we evaluate it)
hist_test_epochs, hist_test_acc = [], []
# π→correctness (store the most recent epoch's bins for plotting/printing)
last_pi_acc_bins, last_pi_bin_counts = None, None
last_pi_bins = (0.25, 0.75)





if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    scaler = None                         # no CUDA ⇒ stay full FP32

elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    scaler = GradScaler()   # <- mixed-precision helper


else:
    DEVICE = torch.device("cpu")
    scaler = None                         




# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed(args.seed)

def pi_stats(model_features, inference_net, loader, device):
    inference_net.eval()
    with torch.no_grad():
        all_pi = []
        for imgs, *_ in loader:
            z = model_features(imgs.to(device, non_blocking=True)).flatten(1)
            all_pi.append(inference_net(z).cpu())
    pi = torch.cat(all_pi)
    mean_pi   = pi.mean().item()
    extreme   = ((pi < 0.25) | (pi > 0.75)).float().mean().item()
    return mean_pi, extreme

@torch.no_grad()
def collect_all_pi(model_features, inference_net, loader, device):
    inference_net.eval()
    out = []
    for imgs, *_ in loader:
        z = model_features(imgs.to(device, non_blocking=True)).flatten(1)
        out.append(inference_net(z).cpu())
    return torch.cat(out, dim=0)  # [N,] or [N,1]


"""# Load datasets for training, validation, and testing
if args.dataset == 'mnist':
    input_channel = 1
    num_classes = 10
    args.n_epoch = 100
    args.batch_size = 64
    args.wd = 1e-3
    args.lr_init = 0.01

    if args.method != 'usdnl':
        Model = LeNet
    else:
        Model = LeNetDO  # with dropout

    train_dataset = data_load.Mnist(root=args.root_dir,
                                    download=True,
                                    train=True,
                                    transform=Model.transform_train,
                                    target_transform=data_tools.transform_target,
                                    dataset=args.dataset,
                                    noise_type=args.noise_type,
                                    =args.noise_ratenoise_rate,
                                    split_per=args.split_percentage,
                                    random_seed=args.seed)

    val_dataset = data_load.Mnist(root=args.root_dir,
                                    train=False,
                                    transform=Model.transform_test,
                                    target_transform=data_tools.transform_target,
                                    dataset=args.dataset,
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate,
                                    split_per=args.split_percentage,
                                    random_seed=args.seed)


    test_dataset = data_load.MnistTest(root=args.root_dir,
                                        transform=Model.transform_test,
                                        target_transform=data_tools.transform_target)


if args.dataset == 'cifar10':
    input_channel = 3
    num_classes = 10
    args.n_epoch = 200
    args.batch_size = 128
    args.wd = 5e-4
    args.lr_init = 0.01

    if args.method != 'usdnl':
        Model = ResNet18
    else:
        Model = ResNet18DO  # with dropout

    train_dataset = data_load.Cifar10(root=args.root_dir,
                                        download=True, 
                                        train=True,
                                        transform=Model.transform_train,
                                        target_transform=data_tools.transform_target,
                                        dataset=args.dataset,
                                        noise_type=args.noise_type,
                                        noise_rate=args.noise_rate,
                                        split_per=args.split_percentage,
                                        random_seed=args.seed)

    val_dataset = data_load.Cifar10(root=args.root_dir,
                                    train=False,
                                    transform=Model.transform_test,
                                    target_transform=data_tools.transform_target,
                                    dataset=args.dataset,
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate,
                                    split_per=args.split_percentage,
                                    random_seed=args.seed)


    test_dataset = data_load.Cifar10Test(root=args.root_dir,
                                            transform=Model.transform_test,
                                            target_transform=data_tools.transform_target)


if args.dataset == 'cifar100':
    input_channel = 3
    num_classes = 100
    args.n_epoch = 200
    args.batch_size = 128
    args.wd = 5e-4
    args.lr_init = 0.01

    if args.method != 'usdnl':
        Model = ResNet34
    else:
        Model = ResNet34DO  # with dropout

    train_dataset = data_load.Cifar100(root=args.root_dir,
                                        download=True,
                                        train=True,
                                        transform=Model.transform_train,
                                        target_transform=data_tools.transform_target,
                                        dataset=args.dataset,
                                        noise_type=args.noise_type,
                                        noise_rate=args.noise_rate,
                                        split_per=args.split_percentage,
                                        random_seed=args.seed)

    val_dataset = data_load.Cifar100(root=args.root_dir,
                                        train=False,
                                        transform=Model.transform_test,
                                        target_transform=data_tools.transform_target,
                                        dataset=args.dataset,
                                        noise_type=args.noise_type,
                                        noise_rate=args.noise_rate,
                                        split_per=args.split_percentage,
                                        random_seed=args.seed)

    test_dataset = data_load.Cifar100Test(root=args.root_dir, 
                                            transform=Model.transform_test, 
                                            target_transform=data_tools.transform_target)"""
# For Food101 dataset (for arlvi and rlvi training):
if args.dataset == "food101":
    input_channel = 3
    num_classes = 101
    # Build a val split only if you are NOT using 100% of noisy train
    BUILD_VAL = (args.eval_val_every > 0) and (0.0 < args.split_percentage < 1.0)




    ###----------------------------------------------
    # Food101 transform data block: 
    ###----------------------------------------------

    # Normalize using per-channel means and stds of imageNet training set
    # ResNet50 was trained on ImageNet with this exact normalization
    # Apply so that inputs aren't out of distribution for the pretrained model

    # ImageNet normalization
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])

    # Common training pipeline for ImageNet-pretrained ResNet on Food-101

    transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0),
                                 interpolation=InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(0.1, 0.1, 0.1, 0.02),
    transforms.ToTensor(),
    # RandomErasing expects a tensor
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'),
    normalize,
])



    # Standard eval pipeline
    transform_test = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])


    data_root = os.path.join(args.root_dir, "food-101")
    if os.path.isdir(data_root):
        args.download = False  # dataset already staged locally

    train_dataset = data_load.Food101(
        root=args.root_dir,
        split="train",
        transform=transform_train,
        split_per=args.split_percentage,
        random_seed=args.seed,
        stratified=True,
        download=args.download
        )

    if BUILD_VAL:
        val_dataset = data_load.Food101(
            root=args.root_dir,
            split="val",
            transform=transform_test,
            split_per=args.split_percentage,
            random_seed=args.seed,
            stratified=True,
            download=args.download
    )
    else:
        val_dataset = None
        # Prevent accidental attempts to evaluate val
        args.eval_val_every = -1

    test_dataset = data_load.Food101(
        root=args.root_dir,
        split="test",
        transform=transform_test,
        split_per=1.0,
        download=args.download
        )

# --------- LMDB-backed Food-101 datasets ----------

"""lmdb_root = os.path.join(args.root_dir, 'food101_lmdb')
train_dataset = data_load.Food101LMDB(os.path.join(lmdb_root,'food101_train.lmdb'),
                                      transform=transform_train)
val_dataset   = data_load.Food101LMDB(os.path.join(lmdb_root,'food101_val.lmdb'),
                                      transform=transform_test)
test_dataset  = data_load.Food101LMDB(os.path.join(lmdb_root,'food101_test.lmdb'),
                                      transform=transform_test)"""


# For alternative methods:
# create rate_schedule to gradually consider less and less samples
if args.forget_rate is None:
    forget_rate = args.noise_rate
    if args.noise_type == 'asymmetric':
        forget_rate /= 2.
else:
    forget_rate = args.forget_rate
num_gradual = min(args.num_gradual, args.n_epoch)
rate_schedule = np.ones(args.n_epoch) * forget_rate
rate_schedule[:num_gradual] = np.linspace(0, forget_rate**args.exponent, num_gradual)



# Prepare a structured output
save_dir = f"{args.result_dir}/{args.dataset}/{args.method}"
if not os.path.exists(save_dir):
    os.system('mkdir -p %s' % save_dir)

model_str = f"{args.dataset}_{args.method}_{args.noise_type}_{args.noise_rate}"
txtfile = f"{save_dir}/{model_str}-s{args.seed}.txt"

# Ensure the results directory exists before writing
if not os.path.exists(os.path.dirname(txtfile)):
    os.makedirs(os.path.dirname(txtfile))

if os.path.exists(txtfile):
    curr_time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    new_dest = f"{txtfile}.bak-{curr_time}"
    os.system(f"mv {txtfile} {new_dest}")

class CombinedModel(torch.nn.Module):
    def __init__(self, features, classifier):
        super().__init__()
        self.features = features
        self.classifier = classifier

    def forward(self, x):
        z = self.features(x)              # shape: [B, 2048, 1, 1]
        z = z.view(z.size(0), -1)         # shape: [B, 2048]
        return self.classifier(z)         # shape: [B, 101]

def group_params_for_wd(module):
    """
    Split params into weight-decay and no-decay groups.
    No decay for biases and normalization layers.
    """
    decay, no_decay = [], []
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        is_bias = name.endswith('.bias')
        in_norm = any(k in name.lower() for k in ['bn', 'batchnorm', 'layernorm', 'groupnorm', 'ln', 'gn'])
        (no_decay if (is_bias or in_norm) else decay).append(p)
    return decay, no_decay

def set_bn_eval(m: torch.nn.Module):
    # Freeze BN running stats but keep gamma/beta trainable
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.eval()

def _save_checkpoint(path, *,
                     model_features, model_classifier, inference_net,
                     optim_all=None, 
                     schedulers=None, epoch:int=0, val_acc:float=float('-inf')):
    to_save = {
        'epoch': epoch,
        'val_acc': val_acc,
        'model_features': model_features.state_dict(),
        'model_classifier': model_classifier.state_dict(),
        'inference_net': inference_net.state_dict(),
    }
    if optim_all is not None:
        to_save['optim_all'] = optim_all.state_dict()   # <-- save under 'optim_all'
    if schedulers is not None:
        to_save['schedulers'] = {k: v.state_dict() for k, v in schedulers.items()}
    torch.save(to_save, path)



def run():
    train_acc = 0.0
    val_acc = 0.0
    test_acc = 0.0
        # Data Loaders
    # Keep this small on Colab until you confirm fast local IO
    workers = args.num_workers


    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=workers,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        persistent_workers=True,  
        prefetch_factor=4       
    )

    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=args.batch_size,
            num_workers=workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4
    )
    else:
        val_loader = None

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=0,              # leave single-process for test; it’s fine
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )




    # Prepare ARLVI*Food101 models and optimizers
    if args.dataset == 'food101' and args.method in ['arlvi_zscore', 'rlvi']:
        # Load pretrained ResNet50 (resnet18 for faster training during testing)
        #backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        in_dim = backbone.fc.in_features                 # <- get it while fc is Linear
        backbone.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(in_dim, num_classes)
        )
        # Split the model into a feature extractor and classifier
        model_features = torch.nn.Sequential(*list(backbone.children())[:-1])
        model_classifier = backbone.fc

        # >>> Freeze BN stats only for A-RLVI. Keep BN trainable for RLVI. <<<
        if args.method in ['arlvi_zscore']:
            model_features.apply(set_bn_eval)

        feature_dim = in_dim  # 2048 for ResNet50, 512 for ResNet18
        

        model = CombinedModel(model_features, model_classifier)

    else:
        model = Model(input_channel=input_channel, num_classes=num_classes)


    model.to(DEVICE)

    # -------------------------------------------------

    #initialize inference network for ARLVI    
    #instantiate the inference network
    inference_net = InferenceNet(feature_dim).to(DEVICE)
    
    # --- optimizers ---
    # --- optimizers (no freezing, both learn from step 0) ---
    # param groups already split by group_params_for_wd(...)
    # === Unified optimizer (AdamW) for backbone, head, and inference net ===
    bb_decay,  bb_no_decay  = group_params_for_wd(model_features)
    hd_decay,  hd_no_decay  = group_params_for_wd(model_classifier)
    inf_params = list(inference_net.parameters())

    param_groups = [
    {'params': bb_decay,    'weight_decay': args.wd_backbone},
    {'params': bb_no_decay, 'weight_decay': 0.0},
    {'params': hd_decay,    'weight_decay': args.wd_head},
    {'params': hd_no_decay, 'weight_decay': 0.0},
    ]
    
    if args.method in ['arlvi_zscore']:
        param_groups.append({'params': inf_params, 'weight_decay': args.wd_inference})

    optim_all = torch.optim.AdamW(param_groups, lr=1e-3)

    # Use a smaller LR for the pretrained backbone, larger LR for the new head.
    lr_bb = args.lr_init * 0.3
    lr_hd = args.lr_init * 3.0

    param_groups = [
        {'params': bb_decay,    'weight_decay': args.wd_backbone, 'lr': lr_bb},
        {'params': bb_no_decay, 'weight_decay': 0.0,              'lr': lr_bb},
        {'params': hd_decay,    'weight_decay': args.wd_head,     'lr': lr_hd},
        {'params': hd_no_decay, 'weight_decay': 0.0,              'lr': lr_hd},
    ]

    if args.method in ['arlvi_zscore']:
        # inference net (only used by A-RLVI(zscore))
        param_groups.append({'params': inf_params, 'weight_decay': args.wd_inference, 'lr': args.lr_inference})

    optim_all = torch.optim.AdamW(param_groups, lr=args.lr_init)



    # pass both to train_arlvi_zscore / train_rlvi as dict
    # ─── unified optimizer ────────────────────────────────
    optimizer = optim_all

    # Define the learning rate scheduler
    # ─── unified LR scheduler ────────────────────────────────
    if args.method in ['arlvi_zscore']:

        steps_per_epoch = len(train_loader)
        # Per-group max LRs map to param group order above:
        # [bb_decay, bb_no_decay, hd_decay, hd_no_decay, inf_params]
        scheduler_all = OneCycleLR(
            optim_all,
            max_lr=[1e-3, 1e-3, 4e-3, 4e-3, 1e-3],
            div_factor=10.0,
            final_div_factor=5e3,
            pct_start=max(0.05, args.warmup_epochs / args.n_epoch),
            steps_per_epoch=steps_per_epoch,
            epochs=args.n_epoch
            )
    else:
        scheduler_all = None


    if args.method == 'jocor':
        model_sec = Model(input_channel=input_channel, num_classes=num_classes)
        model_sec.to(DEVICE)
        optimizer = torch.optim.SGD(
            list(model.parameters()) + list(model_sec.parameters()), 
            lr=args.lr_init, weight_decay=args.wd, momentum=args.momentum
        )


    if args.method == 'coteaching':
        model_sec = Model(input_channel=input_channel, num_classes=num_classes)
        model_sec.to(DEVICE)
        optimizer_sec = torch.optim.SGD(model_sec.parameters(), lr=args.lr_init, weight_decay=args.wd, momentum=args.momentum)
        if args.dataset == 'mnist':
            scheduler_sec = MultiplicativeLR(optimizer_sec, utils.get_lr_factor)
        else:
            scheduler_sec = CosineAnnealingLR(optimizer_sec, T_max=200)
    

    if args.method in ['rlvi', 'arlvi_zscore']:
        sample_weights = torch.ones(len(train_dataset)).to(DEVICE)
        residuals = torch.zeros_like(sample_weights).to(DEVICE)
        overfit = False
        threshold = 0
        val_acc_old, val_acc_old_old = 0, 0


    # Training
    #for early stopping
    best_val = float('-inf')
    epochs_no_improve = 0
    ckpt_path = os.path.join(save_dir, f'best_s{args.seed}.pt')


    # Vector of class-specific priors, initialized to 0.75
    pi_bar_class = torch.full((101,), 0.75, dtype=torch.float32).to(DEVICE)


    for epoch in range(1, args.n_epoch + 1):
        model.train()
        # Keep BN layers (for A-RLVI) using frozen running stats during training (re-apply each epoch)
        if 'model_features' in locals() and args.method in ['arlvi_zscore']:
            model_features.apply(set_bn_eval)

        time_ep = time.time()
        # Reset per-epoch eval placeholders so we never reuse stale values
        val_acc = None
        test_acc = None


        #### Start one epoch of training with selected method ####

        if args.method == "regular":
            train_acc = methods.train_regular(train_loader, model, optimizer)
            val_acc = utils.evaluate(val_loader, model)

        elif args.method == "rlvi":
            start_time = time.time()
            train_acc, threshold, train_loss = methods.train_rlvi(
                train_loader, model, optimizer,
                residuals, sample_weights, overfit, threshold, 
                writer=None, epoch=epoch
            )

            epoch_time = time.time() - start_time
            val_acc = utils.evaluate(val_loader, model)    # computed every epoch
            test_acc = utils.evaluate(test_loader, model)  # computed every epoch

            # Print once with everything
            print(
                f"[rlvi] ep {epoch:03d} | time={epoch_time:.1f}s | "
                f"train={train_acc*100:.2f}% | val={val_acc:.2f}% | test={test_acc:.2f}% | "
                f"π̄={sample_weights.mean().item():.3f}"
            )



        elif args.method == "arlvi_zscore":
            start_time = time.time()
            # --- Train z-score target A-RLVI ---
            ce_loss, kl_loss, train_acc, diag = methods.train_arlvi_zscore(
                model_features         = model_features,
                model_classifier       = model_classifier,
                inference_net          = inference_net,
                dataloader             = train_loader,
                optim_all              = optim_all,       # << one optimizer
                device                 = DEVICE,
                epoch                  = epoch,
                tau                    = args.tau,
                ema_alpha              = args.ema_alpha,
                scheduler              = scheduler_all,   # << one scheduler (or None)
                scaler                 = scaler,
                grad_clip              = None,            # (clip head only; optional)
                return_diag            = True
            )



            # histories + console print
            hist_ce.append(float(ce_loss))
            hist_kl.append(float(kl_loss))
            hist_grad_bb.append(float(diag.get('grad_backbone', 0.0)))
            hist_grad_cls.append(float(diag.get('grad_classifier', 0.0)))
            hist_grad_inf.append(float(diag.get('grad_inference', 0.0)))
            hist_pi_min.append(float(diag.get('pi_min', 0.0)))
            hist_pi_max.append(float(diag.get('pi_max', 0.0)))
            hist_pi_mean.append(float(diag.get('pi_mean', 0.0)))
            
            # LR traces (per batch of this epoch)
            hist_lr_bb.extend(diag.get('lr_trace_backbone', []))
            hist_lr_cls.extend(diag.get('lr_trace_classifier', []))

            # π→correctness (for reporting/plotting)
            last_pi_acc_bins   = diag.get('pi_acc_bins', None)
            last_pi_bin_counts = diag.get('pi_bin_counts', None)


            val_acc  = None
            test_acc = None

            if args.eval_val_every > 0 and epoch % args.eval_val_every == 0:
                val_acc = utils.evaluate(val_loader, model)

            if args.eval_test_every > 0 and epoch % args.eval_test_every == 0:
                test_acc = utils.evaluate(test_loader, model)


            epoch_time = time.time() - start_time


            v = "—" if val_acc is None else f"{val_acc:.2f}%"
            t = "—" if test_acc is None else f"{test_acc:.2f}%"

            print(
                f"[ep {epoch:03d}] | time={epoch_time:.2f}s | "
                f"CE={hist_ce[-1]:.3f} KL={hist_kl[-1]:.3f} | "
                f"∥g∥ bb={hist_grad_bb[-1]:.3f} cls={hist_grad_cls[-1]:.3f} inf={hist_grad_inf[-1]:.3f} | "
                f"π μ={hist_pi_mean[-1]:.3f} min={hist_pi_min[-1]:.2f} max={hist_pi_max[-1]:.2f} | "
                f"train={train_acc*100:.2f}% val={v} test={t}"
            )

            if last_pi_acc_bins is not None:
                lo, hi = diag.get('pi_bins', (0.25, 0.75))
                print(
                f"    π→acc by bin:  "
                f"<{lo:.2f} = {last_pi_acc_bins['lt_0.25']*100:.2f}% (n={last_pi_bin_counts['lt_0.25']}),  "
                f"{lo:.2f}–{hi:.2f} = {last_pi_acc_bins['0.25_0.75']*100:.2f}% (n={last_pi_bin_counts['0.25_0.75']}),  "
                f">{hi:.2f} = {last_pi_acc_bins['gt_0.75']*100:.2f}% (n={last_pi_bin_counts['gt_0.75']})"
            )




        elif args.method == 'coteaching':
            model_sec.train()
            train_acc = methods.train_coteaching(
                train_loader, epoch, 
                model, optimizer, model_sec, optimizer_sec,
                rate_schedule
            )
            val_acc = utils.evaluate(val_loader, model)
            scheduler_sec.step()

        elif args.method == 'jocor':
            model_sec.train()
            train_acc = methods.train_jocor(
                train_loader, epoch, 
                model, model_sec, optimizer, 
                rate_schedule
            )
            val_acc = utils.evaluate(val_loader, model)

        elif args.method == 'cdr':
            train_acc = methods.train_cdr(train_loader, epoch, model, optimizer, rate_schedule)
            val_acc = utils.evaluate(val_loader, model)

        elif args.method == 'usdnl':
            train_acc = methods.train_usdnl(train_loader, epoch, model, optimizer, rate_schedule)
            val_acc = utils.evaluate(val_loader, model)

        elif args.method == 'bare':
            train_acc = methods.train_bare(train_loader, model, optimizer, num_classes)
            val_acc = utils.evaluate(val_loader, model)

        # --- Universal evals (schedule-driven; append exactly once per scheduled epoch) ---
        if args.eval_val_every > 0 and (val_loader is not None) and (epoch % args.eval_val_every == 0):
            if val_acc is None:
                val_acc = utils.evaluate(val_loader, model)

        if args.eval_test_every > 0 and (epoch % args.eval_test_every == 0):
            if test_acc is None:
                test_acc = utils.evaluate(test_loader, model)
            hist_test_epochs.append(epoch)
            hist_test_acc.append(float(test_acc))


        # === EARLY STOPPING  ===
        if args.early_stop and (val_loader is not None):
            # If you already computed val_acc this epoch, reuse it; otherwise compute it now.
            _es_val = val_acc if ('val_acc' in locals() and val_acc is not None) else utils.evaluate(val_loader, model)

            if _es_val > best_val:  # any improvement counts
                best_val = _es_val
                epochs_no_improve = 0
                _save_checkpoint(
                    ckpt_path,
                    model_features=model_features,
                    model_classifier=model_classifier,
                    inference_net=inference_net,
                    optim_all=optim_all,     # unified optimizer
                    schedulers={'all': scheduler_all} if scheduler_all is not None else None,
                    epoch=epoch,
                    val_acc=best_val
                )
                print(f"[ES] New best val acc: {best_val:.2f}% — checkpoint saved.")
            else:
                epochs_no_improve += 1
                print(f"[ES] No improvement ({epochs_no_improve}/{args.early_stop_patience}). "
                    f"Best={best_val:.2f}%")

                if epochs_no_improve >= args.early_stop_patience:
                    print("[ES] Patience exhausted. Stopping training early.")
                    break
        # === END EARLY STOPPING ===


        #### Finish one epoch of training with selected method ####

        # Log info
        time_ep = time.time() - time_ep
        #test_acc = utils.evaluate(test_loader, model)

        # Print log-table
        if (epoch + 1) % args.print_freq == 0 and (test_acc is not None):
            utils.output_table(epoch, args.n_epoch, time_ep, train_acc=train_acc, test_acc=test_acc)
        else:
            utils.output_table(epoch, args.n_epoch, time_ep, train_acc=train_acc)


        # Prepare output: put dummy values for alternative methods
        if args.method != 'rlvi':
            overfit = False
            threshold = 0
            clean, corr = 1, 0
        # Check the number of correctly identified corrupted samples for RLVI
        if args.method == 'rlvi' and args.noise_rate > 0 and hasattr(train_dataset, 'noise_mask'):
            mask = (sample_weights > threshold).cpu()
            clean, corr = utils.get_ratio_corrupted(mask, train_dataset.noise_mask)
        else:
            clean, corr = 1, 0

        # Save logs to the file
        """with open(txtfile, "a") as myfile:
            myfile.write(f"{int(epoch)}:\t{time_ep:.2f}\t{threshold:.2f}\t{overfit}\t"
                         + f"{clean*100:.2f}\t{corr*100:.2f}\t"
                         + f"{train_acc:8.4f}\t{val_acc:8.4f}\t{test_acc:8.4f}\n")
"""

        # Restore best checkpoint if we used early stopping
    if args.early_stop and os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location=DEVICE)
        model_features.load_state_dict(state['model_features'])
        model_classifier.load_state_dict(state['model_classifier'])
        inference_net.load_state_dict(state['inference_net'])

        if 'optim_all' in state:
            optim_all.load_state_dict(state['optim_all'])

        if 'schedulers' in state and scheduler_all is not None and 'all' in state['schedulers']:
            scheduler_all.load_state_dict(state['schedulers']['all'])

        print(f"[ES] Restored best model from epoch {state.get('epoch','?')} "
            f"with val acc {state.get('val_acc',0):.2f}%")


    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=DEVICE)
        model_features.load_state_dict(state['model_features'])
        model_classifier.load_state_dict(state['model_classifier'])
        inference_net.load_state_dict(state['inference_net'])
        if 'optim_all' in state:
            optim_all.load_state_dict(state['optim_all'])
        if 'schedulers' in state and scheduler_all is not None and 'all' in state['schedulers']:
            scheduler_all.load_state_dict(state['schedulers']['all'])
        final_test = utils.evaluate(test_loader, model)
        print(f"Loaded best checkpoint (epoch {state['epoch']}, val={state['val_acc']:.2f}%) → test={final_test:.2f}%")

    plot_dir = os.path.join(save_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Losses
    plt.figure()
    plt.plot(hist_ce, label='CE (avg/epoch)')
    plt.plot(hist_kl, label='KL (avg/epoch)')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('CE & KL'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'losses_ce_kl.png'), dpi=150); plt.close()

    # Grad norms
    plt.figure()
    plt.plot(hist_grad_bb,  label='Backbone ∥g∥')
    plt.plot(hist_grad_cls, label='Classifier ∥g∥')
    plt.plot(hist_grad_inf, label='Inference ∥g∥')
    plt.xlabel('Epoch'); plt.ylabel('RMS grad norm'); plt.title('Gradients'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'grad_norms.png'), dpi=150); plt.close()

    # π histogram (final)
    if args.method == 'arlvi_zscore' and 'model_features' in locals():
        pi_all = collect_all_pi(model_features, inference_net, train_loader, DEVICE).flatten().cpu().numpy()
        plt.figure()
        plt.hist(pi_all, bins=50, range=(0.0, 1.0))
        plt.xlabel('π')
        plt.ylabel('Count')
        plt.title(f'π histogram (final) — {args.method}')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'pi_histogram_{args.method}.png'), dpi=150)
        plt.close()

    elif args.method == 'rlvi':
        # sample_weights is the RLVI π vector
        pi_all = sample_weights.detach().cpu().numpy()
        plt.figure()
        plt.hist(pi_all, bins=50, range=(0.0, 1.0))
        plt.xlabel('π')
        plt.ylabel('Count')
        plt.title('π histogram (final) — RLVI')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'pi_histogram_rlvi.png'), dpi=150)
        plt.close()


        # LR traces across training (per batch)
    if len(hist_lr_bb) and len(hist_lr_cls):
        plt.figure()
        plt.plot(hist_lr_bb, label='Backbone LR')
        plt.plot(hist_lr_cls, label='Head LR')
        plt.xlabel('Training step'); plt.ylabel('LR'); plt.title('OneCycle LR traces'); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'lr_traces.png'), dpi=150); plt.close()

    # π → correctness bars (last epoch's bins)
    if last_pi_acc_bins is not None:
        lo, hi = 0.25, 0.75
        plt.figure()
        bins = [f'<{lo:.2f}', f'{lo:.2f}–{hi:.2f}', f'>{hi:.2f}']
        vals = [last_pi_acc_bins['lt_0.25']*100.0,
                last_pi_acc_bins['0.25_0.75']*100.0,
                last_pi_acc_bins['gt_0.75']*100.0]

        plt.bar(bins, vals)
        plt.ylim(0, 100)
        plt.ylabel('Accuracy (%)'); plt.title('π → correctness (last epoch)'); plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'pi_to_correctness.png'), dpi=150); plt.close()
    

    # If no scheduled test eval happened, add a final point so a plot is produced
    if not len(hist_test_acc):
        final_test = utils.evaluate(test_loader, model)
        hist_test_epochs.append(epoch)  # last completed epoch index
        hist_test_acc.append(float(final_test))

    # Test accuracy over epochs (only at evaluation epochs)
    if len(hist_test_acc):
        plt.figure()
        # Connect the sparse points with a line; x-axis is the true epoch index
        plt.plot(hist_test_epochs, hist_test_acc, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Test accuracy (%)')
        plt.title('Test accuracy (evaluated every eval_test_every epochs)')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'test_accuracy_over_epochs.png'), dpi=150)
        plt.close()

        # === Save this run's test-acc history and make overlays ===
    import glob, numpy as _np

    hist_dir = os.path.join(save_dir, "histories")
    os.makedirs(hist_dir, exist_ok=True)

    # Build a readable label for overlays
    try:
        arch = backbone.__class__.__name__  # e.g., ResNet50
    except NameError:
        arch = model.__class__.__name__     # fallback (e.g., CombinedModel or ResNet18)

    run_label = f"{args.method}-{arch}-seed{args.seed}"
    # Include tau if present (A-RLVI variants)
    if hasattr(args, 'tau'):
        run_label += f"-tau{args.tau}"

    # Persist this run's curve
    _np.savez(
        os.path.join(hist_dir, f"test_acc_{run_label}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.npz"),
        epochs=_np.array(hist_test_epochs, dtype=_np.int32),
        acc=_np.array(hist_test_acc, dtype=_np.float32),
        label=run_label,
    )

    # Overlay: all runs in THIS method folder
    method_overlay = os.path.join(plot_dir, "test_accuracy_over_runs.png")
    files = sorted(glob.glob(os.path.join(hist_dir, "test_acc_*.npz")))
    if files:
        plt.figure()
        for f in files:
            d = _np.load(f, allow_pickle=False)
            lbl = str(d["label"]) if "label" in d.files else os.path.basename(f)
            plt.plot(d["epochs"], d["acc"], marker='o', linewidth=1.5, label=lbl)
        plt.xlabel('Epoch'); plt.ylabel('Test accuracy (%)')
        plt.title(f'Test accuracy — {args.dataset}/{args.method}')
        plt.legend(loc='best', fontsize=8)
        plt.tight_layout()
        plt.savefig(method_overlay, dpi=150); plt.close()

    # Overlay: all methods for THIS dataset
    dataset_hist_glob = os.path.join(args.result_dir, args.dataset, "*", "histories", "test_acc_*.npz")
    files = sorted(glob.glob(dataset_hist_glob))
    if files:
        combined_dir = os.path.join(args.result_dir, args.dataset, "_combined_plots")
        os.makedirs(combined_dir, exist_ok=True)
        out_path = os.path.join(combined_dir, "test_accuracy_over_methods.png")
        plt.figure()
        for f in files:
            d = _np.load(f, allow_pickle=False)
            lbl = str(d["label"]) if "label" in d.files else os.path.basename(f)
            plt.plot(d["epochs"], d["acc"], marker='o', linewidth=1.5, label=lbl)
        plt.xlabel('Epoch'); plt.ylabel('Test accuracy (%)')
        plt.title(f'Test accuracy — all methods ({args.dataset})')
        plt.legend(loc='best', fontsize=8)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150); plt.close()


    print(f"Saved plots to: {plot_dir}")


if __name__ == '__main__':
    run()
