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









parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', type=str, help = 'dir to save result txt files', default='results/')
parser.add_argument('--root_dir', type=str, help = 'dir that stores datasets', default='data/')
parser.add_argument('--dataset', type=str, help='[mnist, cifar10, cifar100, food101]', default='mnist')
parser.add_argument('--method', type=str, help='[regular, rlvi, arlvi, arlvi_vanilla, coteaching, jocor, cdr, usdnl, bare]', default='rlvi')

###---for A-RLVI stabilization---###
parser.add_argument('--update_inference_every', type=str, choices=['batch', 'epoch'], default='batch')
parser.add_argument('--beta', type=float, default=1.0,
                    help='Weight for the KL divergence regularization term (vanilla A-RLVI)')

parser.add_argument('--warmup_epochs', type=int, default=2,
                    help='Number of warm-up epochs where π̄ is fixed (default: 2)')
parser.add_argument('--ema_alpha', type=float, help='momentum in ema average', default=0.90)
parser.add_argument('--beta_entropy_reg', type=float, help='coefficient for entropy regularization strength', default=0.05)
parser.add_argument('--lr_inference', type=float, default=5e-5, help='Learning rate for the inference network (Adam)')
parser.add_argument('--lr_init', type=float, default=1e-3,
                    help='Initial learning rate for model (used by SGD)')
parser.add_argument('--split_percentage', type=float, default=0.1)
###---###
parser.add_argument('--n_epoch', type=int, help='number of epochs for training', default=80)
parser.add_argument('--batch_size', type=int, help='batch size for training', default=64)

parser.add_argument('--wd', type=float, help='Weight decay for optimizer', default=1e-4)
parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.25)
parser.add_argument('--noise_type', type=str, help='[pairflip, symmetric, asymmetric, instance]', default='pairflip')
parser.add_argument('--momentum', type=int, help='momentum', default=0.9)
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
if DEVICE == "cuda":
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
    extreme   = ((pi < 0.2) | (pi > 0.8)).float().mean().item()
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

    ###----------------------------------------------
    # Food101 transform data block: 
    ###----------------------------------------------

    # Normalize using per-channel means and stds of imageNet training set
    # ResNet50 was trained on ImageNet with this exact normalization
    # Apply so that inputs aren't out of distribution for the pretrained model
    normalize = transforms.Normalize([0.485,0.456,0.406],
                                    [0.229,0.224,0.225])

    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])


    train_dataset = data_load.Food101(
        root=args.root_dir,
        split="train",
        transform=transform_train,
        split_per=args.split_percentage,
        random_seed=args.seed,
        download=True
    )
    val_dataset = data_load.Food101(
        root=args.root_dir,
        split="val",
        transform=transform_test,
        split_per=args.split_percentage,
        random_seed=args.seed,
        download=True
    )
    test_dataset = data_load.Food101(
        root=args.root_dir,
        split="test",          # clean!
        transform=transform_test,
        split_per=1.0,
        download=True
    )



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


def run():
    train_acc = 0.0
    val_acc = 0.0
    test_acc = 0.0
    # Data Loaders
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               drop_last=False,
                                               shuffle=True,
                                               pin_memory=True,
                                               prefetch_factor=2)
    
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                            batch_size=args.batch_size,
                                            num_workers=args.num_workers,
                                            drop_last=False,
                                            shuffle=False,
                                            pin_memory=True,
                                            prefetch_factor=2)

    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=args.num_workers,
                                              drop_last=False,
                                              shuffle=False,
                                              pin_memory=True,
                                              prefetch_factor=4)

    # Prepare ARLVI*Food101 models and optimizers
    if args.dataset == 'food101' and args.method in ['arlvi', 'arlvi_vanilla']:
        # Load pretrained ResNet50 (resnet18 for faster training during testing)
        #backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        backbone.fc = torch.nn.Linear(backbone.fc.in_features, num_classes) #final classification layer
        # Split the model into a feature extractor and classifier
        model_features = torch.nn.Sequential(*list(backbone.children())[:-1])
        model_classifier = backbone.fc

        model = CombinedModel(model_features, model_classifier)

    else:
        model = Model(input_channel=input_channel, num_classes=num_classes)


    model.to(DEVICE)

    # -------------------------------------------------
    # split params
    backbone_params   = list(model_features.parameters())
    classifier_params = list(model_classifier.parameters())

    # Optimize backbone and classifier separately - backbone is SGD, classifier is AdamW
    
    #Adam gives the “still learning” head its own adaptive step-size 
    #to cope with noisy, rapidly changing sample weights, while SGD 
    #keeps the convolutional filters safe from destructive updates.
    optim_backbone = torch.optim.SGD(
            backbone_params, 
            lr=args.lr_init, 
            momentum=args.momentum, 
            weight_decay=args.wd
        )
    # AdamW optimizer for the classifier so that it can adapt quickly to non-stationary loss due to changing pi_i values
    # weight noise makes effective LR fluctuate → SGD can stall
    # Adam’s normalisation smooths those fluctuations
    optim_classifier = torch.optim.AdamW(
            classifier_params, 
            lr=args.lr_init * 10,  # 10x larger than backbone
            weight_decay=5e-4
        )

    # pass both to train_arlvi as dict
    # ─── unified optimizer ────────────────────────────────
    optimizer = {"backbone": optim_backbone, "classifier": optim_classifier}

    # Define the learning rate scheduler
    # ─── unified LR scheduler ────────────────────────────────
    if args.method in ['arlvi', 'arlvi_vanilla']:
        # total number of batches per epoch:
        steps_per_epoch = len(train_loader)

        scheduler_backbone = OneCycleLR(
                optim_backbone,
                max_lr=args.lr_init * 5,      
                div_factor=10.,      
                final_div_factor=1e4,
                pct_start=args.warmup_epochs / args.n_epoch, 
                steps_per_epoch=steps_per_epoch, 
                epochs=args.n_epoch
            )

        scheduler_classifier = OneCycleLR(
                optim_classifier,
                max_lr=args.lr_init*10,      
                div_factor=10.,     
                final_div_factor=1e4,
                pct_start=args.warmup_epochs / args.n_epoch, 
                steps_per_epoch=steps_per_epoch, 
                epochs=args.n_epoch
            )
        schedulers = {"backbone": scheduler_backbone,
              "classifier": scheduler_classifier}



    else:
        # MNIST/CIFAR fallback
        if args.dataset == 'mnist':
            scheduler = MultiplicativeLR(optimizer, utils.get_lr_factor)
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=200)


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
    

    if args.method in ['rlvi', 'arlvi']:
        sample_weights = torch.ones(len(train_dataset)).to(DEVICE)
        residuals = torch.zeros_like(sample_weights).to(DEVICE)
        overfit = False
        threshold = 0
        val_acc_old, val_acc_old_old = 0, 0


    # Evaluate init model
    test_acc = utils.evaluate(test_loader, model)
    utils.output_table(epoch=0, n_epoch=args.n_epoch, test_acc=test_acc)
    with open(txtfile, "a") as myfile:
        myfile.write("epoch:\ttime_ep\ttau\tfix\tclean,%\tcorr,%\ttrain_acc\tval_acc\ttest_acc\n")
        myfile.write(f"0:\t0\t0\t{False}\t100\t0\t0\t0\t{test_acc:8.4f}\n")

 
    #initialize inference network for ARLVI
    # Get the correct feature dimension from the backbone
    feature_dim = model_classifier.in_features   # 512 (R-18) / 2048 (R-50)

    # Instantiate the inference network with that dimension
    inference_net = InferenceNet(feature_dim).to(DEVICE)

    # Optimiser for the inference network
    optimizer_inf = torch.optim.Adam(inference_net.parameters(),
                                    lr=args.lr_inference, weight_decay=1e-4)

    # Training
    # Vector of class-specific priors, initialized to 0.75
    pi_bar_class = torch.full((101,), 0.75, dtype=torch.float32).to(DEVICE)


    for epoch in range(1, args.n_epoch):
        model.train()

        time_ep = time.time()

        #### Start one epoch of training with selected method ####

        if args.method == "regular":
            train_acc = methods.train_regular(train_loader, model, optimizer)
            val_acc = utils.evaluate(val_loader, model)

        elif args.method == "rlvi":
            start_time = time.time()
            # --- Train RLVI ---
            train_acc, threshold, train_loss = methods.train_rlvi(
            train_loader, model, optimizer,
            residuals, sample_weights, overfit, threshold, 
            writer=None, epoch=epoch
            )

            epoch_time = time.time() - start_time
            val_acc = utils.evaluate(val_loader, model)
            test_acc = utils.evaluate(test_loader, model)
            
            
        elif args.method == "arlvi":
            # --- Train ARLVI ---
            start_time = time.time()
            ce_loss, kl_loss, train_acc, pi_bar_class = methods.train_arlvi(
                    scaler              = scaler,                 # mixed-precision scaler
                    model_features      = model_features,
                    model_classifier    = model_classifier,
                    inference_net       = inference_net,
                    dataloader          = train_loader,
                    optimizer           = optimizer,
                    inference_optimizer = optimizer_inf,          # ← renamed
                    device              = DEVICE,
                    epoch               = epoch,
                    lambda_kl           = args.lambda_kl,
                    pi_bar              = 0.75,                   # scalar warm-up prior
                    warmup_epochs       = args.warmup_epochs,
                    alpha               = args.ema_alpha,
                    pi_bar_class        = pi_bar_class,           # running tensor
                    beta                = args.beta_entropy_reg,
                    tau                 = 1,                    # leave default
                    scheduler           = schedulers,             # ← pass the dict
                    grad_clip           = 5.0,
            )
            
            


            # Check if overfitting has started
            if not overfit:
                if epoch > 2:
                    # Overfitting started <=> validation score is dropping
                    overfit = (val_acc < 0.5 * (val_acc_old + val_acc_old_old))
                val_acc_old_old = val_acc_old
                val_acc_old = val_acc

        elif args.method == "arlvi_vanilla":
            start_time = time.time()

            ce_loss, kl_loss, train_acc, diag = methods.train_arlvi_vanilla(
                model_features         = model_features,
                model_classifier       = model_classifier,
                inference_net          = inference_net,
                dataloader             = train_loader,
                optim_backbone         = optim_backbone,
                optim_classifier       = optim_classifier,
                optim_inference        = optimizer_inf,
                device                 = DEVICE,
                epoch                  = epoch,
                beta                   = args.beta,
                update_inference_every = args.update_inference_every,
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

            train_acc = utils.get_accuracy(train_loader, model)
            val_acc = utils.evaluate(val_loader, model)

            print(
                f"[ep {epoch:03d}] "
                f"CE={hist_ce[-1]:.3f} KL={hist_kl[-1]:.3f} | "
                f"∥g∥ bb={hist_grad_bb[-1]:.3f} cls={hist_grad_cls[-1]:.3f} inf={hist_grad_inf[-1]:.3f} | "
                f"π μ={hist_pi_mean[-1]:.3f} min={hist_pi_min[-1]:.2f} max={hist_pi_max[-1]:.2f} | "
                f"train_acc={train_acc:.2f}% val_acc={val_acc:.2f}%"
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


        #### Finish one epoch of training with selected method ####

        # Log info
        time_ep = time.time() - time_ep
        #test_acc = utils.evaluate(test_loader, model)

        # Print log-table
        if (epoch + 1) % args.print_freq == 0:
            utils.output_table(epoch, args.n_epoch, time_ep, train_acc=train_acc, test_acc=test_acc)
        else:
            utils.output_table(epoch, args.n_epoch, time_ep, train_acc=train_acc)

        # Prepare output: put dummy values for alternative methods
        if args.method != 'rlvi':
            overfit = False
            threshold = 0
            clean, corr = 1, 0
        # Check the number of correctly identified corrupted samples for RLVI
        if (args.method == 'rlvi') and (args.noise_rate > 0):
            mask = (sample_weights > threshold).cpu()
            clean, corr = utils.get_ratio_corrupted(mask, train_dataset.noise_mask)

        # Save logs to the file
        with open(txtfile, "a") as myfile:
            myfile.write(f"{int(epoch)}:\t{time_ep:.2f}\t{threshold:.2f}\t{overfit}\t"
                         + f"{clean*100:.2f}\t{corr*100:.2f}\t"
                         + f"{train_acc:8.4f}\t{val_acc:8.4f}\t{test_acc:8.4f}\n")

    # <<< ADD THIS WHOLE BLOCK AFTER THE FOR-LOOP, STILL INSIDE run() >>>
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
    if args.method in ['arlvi', 'arlvi_vanilla']:
        pi_all = collect_all_pi(model_features, inference_net, train_loader, DEVICE).flatten().cpu().numpy()
        plt.figure()
        plt.hist(pi_all, bins=50, range=(0.0, 1.0))
        plt.xlabel('π'); plt.ylabel('Count'); plt.title('π histogram (final)'); plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'pi_histogram.png'), dpi=150); plt.close()

    print(f"Saved plots to: {plot_dir}")


if __name__ == '__main__':
    run()
