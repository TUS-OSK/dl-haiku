#!/usr/bin/env python3
import argparse
import json
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from fastprogress import master_bar, progress_bar
from fastprogress.fastprogress import ConsoleMasterBar
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

mb: ConsoleMasterBar
writer: SummaryWriter


def train(args: argparse.Namespace, model: nn.Module, device: torch.device, train_loader: DataLoader,
          optimizer: Optimizer, epoch: int) -> None:
    model.train()
    for batch_idx, (data, target) in enumerate(progress_bar(train_loader, parent=mb)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = F.nll_loss(output, target)

        loss.backward()
        optimizer.step()

        # Report
        n_iter = (epoch - 1) * len(train_loader) + batch_idx
        with torch.no_grad():
            writer.add_scalar('train/loss', loss.item(), n_iter)


def test(args: argparse.Namespace, model: nn.Module, device: torch.device, test_loader: DataLoader,
         epoch: int) -> None:
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in progress_bar(test_loader, parent=mb):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        # Report
        test_loss /= len(test_loader.dataset)
        writer.add_scalar("val/loss", test_loss, epoch)


def main() -> None:
    # Training settings
    parser = argparse.ArgumentParser(description='dl-haiku')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='学習時のバッチサイズ (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='テスト時のバッチサイズ (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='エポック数 (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='CUDAを用いない')
    parser.add_argument('--not-save-model', action='store_false', default=True,
                        help='Modelを保存しない')
    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))

    # seed固定
    torch.manual_seed(args.seed)
    np.random.seed(0)
    random.seed(0)

    # modelの設定
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Datasetの設定
    train_loader = DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # model, optimizerの用意
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters())

    # toolsの用意
    global writer, mb
    writer = SummaryWriter()
    mb = master_bar(range(1, args.epochs + 1))

    # 学習
    for epoch in mb:
        train(args, model, device, train_loader, optimizer, epoch)
        if args.save_model:  # Modelの保存
            torch.save(model.state_dict(), f"model_{epoch}.pt")
        test(args, model, device, test_loader, epoch)
        mb.write(f'Finished loop {epoch}.')
    writer.close()


if __name__ == "__main__":
    main()
