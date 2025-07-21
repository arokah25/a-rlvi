import os
import argparse
from datetime import datetime
import time

import numpy as np
import torch

from torch.optim.lr_scheduler import (
    MultiplicativeLR,      # for MNIST fallback
    CosineAnnealingLR,     # for CIFAR / fallback
    LinearLR,              # warm-up
    SequentialLR           # warm-up → cosine
)

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
from amortized.inference_net import InferenceNet


parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', type=str, help='dir to save result txt files', default='results/')
parser.add_argument('--root_dir', type=str, help='dir that stores datasets', default='data/')
parser.add_argument('--dataset', type=str, help='[mnist, cifar10, cifar100, food101]', default='mnist')
parser.add_argument('--method', type=str, help='[regular, rlvi, arlvi, coteaching, jocor, cdr, usdnl, bare]', default='rlvi')

###---for A-RLVI stabilization---###
parser.add_argument('--lambda_kl', type=float, default=1.0,
                    help='Weight for the KL divergence regularization term')
parser.add_argument('--warmup_epochs', type=int, default=2,
                    help='Number of warm-up epochs where π̄ is fixed (default: 2)')
parser.add_argument('--ema_alpha', type=float, help='momentum in ema average', default=0.95)
parser.add_argument('--beta_entropy_reg', type=float, help='coefficient for entropy regularization strength', default=0.05)
parser.add_argument('--lr_inference', type=float, default=1e-3, help='Learning rate for the inference network (Adam)')
parser.add_argument('--lr_init', type=float, default=0.01,
                    help='Initial learning rate for model (used by SGD)')
parser.add_argument('--split_percentage', type=float, default=0.1)
###---###
parser.add_argument('--n_epoch', type=int, help='number of epochs for training', default=80)  # FIXED: was --n_epochs
parser.add_argument('--batch_size', type=int, help='batch size for training', default=64)

parser.add_argument('--wd', type=float, help='Weight decay for optimizer', default=1e-4)
parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.1)
parser.add_argument('--noise_type', type=str, help='[pairflip, symmetric, asymmetric, instance]', default='pairflip')
parser.add_argument('--momentum', type=int, help='momentum', default=0.9)
parser.add_argument('--print_freq', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=4, help='number of subprocesses for data loading')
parser.add_argument('--seed', type=int, default=1)
# For alternative methods
parser.add_argument('--forget_rate', type=float, help='forget rate', default=None)
parser.add_argument('--num_gradual', type=int, default=10,
                    help='how many epochs for linear drop rate...')
parser.add_argument('--exponent', type=float, default=1,
                    help='exponent of the forget rate...')
parser.add_argument('--debug', action='store_true',
                    help='Print debugging information during training')

args = parser.parse_args()

# TensorBoard logging (for RLVI and ARLVI)
from torch.utils.tensorboard import SummaryWriter
log_dir = f"runs/{args.method}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
writer = SummaryWriter(log_dir=log_dir)

# Device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if DEVICE == "cuda":
    torch.cuda.manual_seed(args.seed)


# ------------------------------
# (Your dataset‐loading blocks here; unchanged except for typo fix
#  in the commented‐out MNIST section: `noise_rate=args.noise_rate`)
# ------------------------------


class CombinedModel(torch.nn.Module):
    def __init__(self, features, classifier):
        super().__init__()
        self.features = features
        self.classifier = classifier

    def forward(self, x):
        z = self.features(x)              # shape: [B, C, 1, 1]
        z = z.view(z.size(0), -1)         # shape: [B, C]
        return self.classifier(z)         # shape: [B, num_classes]


def run():
    # DataLoaders...
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        prefetch_factor=4
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=4
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=4
    )

    # Model + optimizer
    if args.dataset == 'food101' and args.method == 'arlvi':
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        backbone.fc = torch.nn.Linear(backbone.fc.in_features, num_classes)
        model_features = torch.nn.Sequential(*list(backbone.children())[:-1])
        model_classifier = backbone.fc
        model = CombinedModel(model_features, model_classifier)
    else:
        model = Model(input_channel=input_channel, num_classes=num_classes)

    model.to(DEVICE)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr_init,
        weight_decay=args.wd,
        momentum=args.momentum
    )

    # ADDED: unified scheduler setup
    if args.method == 'arlvi':
        # warm-up: 0.1→1.0 over `warmup_epochs`
        warmup_sched = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=args.warmup_epochs
        )
        # cosine: decay from 1.0→eta_min over remaining epochs
        cosine_sched = CosineAnnealingLR(
            optimizer,
            T_max=max(args.n_epoch - args.warmup_epochs, 1),
            eta_min=1e-5
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[args.warmup_epochs]
        )
    else:
        # DELETED: the old, duplicate scheduler definitions
        # ADDED: fallback to your original rules for non-ARLVI
        if args.dataset == 'mnist':
            scheduler = MultiplicativeLR(optimizer, utils.get_lr_factor)
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=200)

    # Instantiate inference net if needed
    if args.method == 'arlvi':
        feature_dim = model_classifier.in_features
        inference_net = InferenceNet(feature_dim).to(DEVICE)
        optimizer_inf = torch.optim.Adam(
            inference_net.parameters(),
            lr=args.lr_inference,
            weight_decay=1e-4
        )

    # Evaluate init model
    test_acc = utils.evaluate(test_loader, model)
    utils.output_table(epoch=0, n_epoch=args.n_epoch, test_acc=test_acc)
    with open(txtfile, "a") as f:
        f.write("epoch:\ttime_ep\t…\ttest_acc\n")
        f.write(f"0:\t0\t…\t{test_acc:8.4f}\n")

    # initialize π̄ₑₘₐ
    pi_bar_ema = 1.0 - args.noise_rate

    # ---------------------
    # Training loop
    # ---------------------
    for epoch in range(1, args.n_epoch):
        model.train()
        t0 = time.time()

        if args.method == "arlvi":
            avg_ce, avg_kl, train_acc, mean_pi, pi_bar_ema = methods.train_arlvi(
                model_features=model_features,
                model_classifier=model_classifier,
                inference_net=inference_net,
                dataloader=train_loader,
                optimizer=optimizer,
                inference_optimizer=optimizer_inf,
                device=DEVICE,
                epoch=epoch,
                lambda_kl=args.lambda_kl,
                warmup_epochs=args.warmup_epochs,
                pi_bar=1.0 - args.noise_rate,
                alpha=args.ema_alpha,
                pi_bar_ema=pi_bar_ema,
                beta=args.beta_entropy_reg,
                writer=writer
            )
            val_acc = utils.evaluate(val_loader, model)

            # Log ARLVI metrics
            writer.add_scalar("Loss/CE_weighted", avg_ce, epoch)
            writer.add_scalar("Loss/KL", avg_kl, epoch)
            writer.add_scalar("Inference/MeanPi", mean_pi, epoch)
            writer.add_scalar("Inference/pi_bar_ema", pi_bar_ema, epoch)
            writer.add_scalar("Train/Accuracy", train_acc, epoch)
            writer.add_scalar("Val/Accuracy", val_acc, epoch)

        else:
            # ... your other `elif args.method == ...` branches unchanged ...
            pass

        # single scheduler step for *all* methods
        scheduler.step()

        # end‐of‐epoch logging
        epoch_time = time.time() - t0
        test_acc = utils.evaluate(test_loader, model)
        utils.output_table(epoch, args.n_epoch, epoch_time,
                           train_acc=train_acc, test_acc=test_acc)

        with open(txtfile, "a") as f:
            f.write(f"{epoch}:\t{epoch_time:.2f}\t…\t{test_acc:8.4f}\n")

    writer.close()


if __name__ == '__main__':
    run()
