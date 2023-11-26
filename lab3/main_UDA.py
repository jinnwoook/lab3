import os
import logging
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

from torch.utils.data import DataLoader
from torchmetrics import Accuracy

from lib.datasets import Cifar10Labeled, Cifar10Unlabeled, CIFAR10_MEAN, CIFAR10_STD
from lib.models import ResNet
from lib.engines import train_one_epoch_semi_supervised, eval_one_epoch
from lib.utils import save_model
from lib.losses import consistency_loss


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--title', default='UDA', type=str)
    parser.add_argument('--train_data', default='data/train', type=str)
    parser.add_argument('--test_data', default='data/test', type=str)
    parser.add_argument('--train_db', default='data/train_labeled.json', type=str)
    parser.add_argument('--train_unlabled_db', default='data/train_unlabeled.json', type=str)
    parser.add_argument('--test_db', default='data/test.json', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--unlabeled_batch_size', default=224, type=int)
    parser.add_argument('--blocks', nargs='+', default=[3, 3, 9, 3], type=int)
    parser.add_argument('--dims', nargs='+', default=[64, 128, 256, 512], type=int)
    parser.add_argument('--dropout', default=0.1, type=float)    
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--epochs', default=2000, type=int)
    parser.add_argument('--lr', default=5e-4, type=int)
    parser.add_argument('--min_lr', default=1e-6, type=int)
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--checkpoint_dir', default='checkpoints', type=str)
    args = parser.parse_args()
    return args


def main(args):
    # -------------------------------------------------------------------------
    # Set Logger & Checkpoint Dirs
    # -------------------------------------------------------------------------
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logging.basicConfig(
        filename=f'{args.title}.log',
        format='%(asctime)s - %(message)s',
        level=logging.INFO, 
    )
    
    # -------------------------------------------------------------------------
    # Data Processing Pipeline
    # -------------------------------------------------------------------------
    train_transform = T.Compose([
        T.RandomCrop(size=32, padding=4),
        T.RandomHorizontalFlip(p=0.5),
        T.RandAugment(num_ops=2, magnitude=9),
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float),
        T.RandomErasing(p=0.25),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_data = Cifar10Labeled(args.train_data, args.train_db, train_transform)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2)

    train_uda_transform = T.Compose([
        T.RandAugment(num_ops=2, magnitude=9),
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_uda_target_transform = T.Compose([
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_uda_data = Cifar10Unlabeled(args.train_data, args.train_unlabled_db, train_uda_transform, train_uda_target_transform)
    train_uda_loader = DataLoader(train_uda_data, batch_size=args.unlabeled_batch_size, shuffle=True, drop_last=True, num_workers=8)


    val_transform = T.Compose([
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    val_data = Cifar10Labeled(args.test_data, args.test_db, transform=val_transform)
    val_loader = DataLoader(val_data, batch_size=256, num_workers=4)

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    model = ResNet(args.blocks, args.dims, args.dropout, args.num_classes)
    model = model.to(args.device)

    # -------------------------------------------------------------------------
    # Performance Metic, Loss Function, Optimizer
    # -------------------------------------------------------------------------
    metric_fn = Accuracy(task='multiclass', num_classes=args.num_classes)
    metric_fn = metric_fn.to(args.device)
    loss_fn = nn.CrossEntropyLoss()
    loss_uda_fn = consistency_loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))

    # -------------------------------------------------------------------------
    # Run Main Loop
    # -------------------------------------------------------------------------
    for epoch in range(args.epochs):
        train_summary = train_one_epoch_semi_supervised(
            model, train_loader, train_uda_loader, metric_fn, loss_fn, loss_uda_fn, args.device,
            optimizer, scheduler)

        # evaluate one epoch
        if (epoch + 1) % 10 == 0:
            val_summary = eval_one_epoch(
                model, val_loader, metric_fn, loss_fn, args.device
            )
            log = (f'epoch {epoch+1}, '
                   + f'train_loss: {train_summary["loss"]:.4f}, '
                   + f'train_accuracy: {train_summary["accuracy"]:.4f}, '
                   + f'val_loss: {val_summary["loss"]:.4f}, '
                   + f'val_accuracy: {val_summary["accuracy"]:.4f}')
            print(log)
            logging.info(log)

        # save model
        checkpoint_path = f'{args.checkpoint_dir}/{args.title}_last.pt'
        save_model(checkpoint_path, model, optimizer, scheduler, epoch+1)


if __name__=="__main__":
    args = get_args()
    main(args)