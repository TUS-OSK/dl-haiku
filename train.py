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

from dataset import SimplifiedDataset
from models import SimplifiedNet

mb: ConsoleMasterBar
writer: SummaryWriter


def train(args: argparse.Namespace, model: nn.Module, device: torch.device, train_loader: DataLoader,
          optimizer: Optimizer, epoch: int) -> None:
    model.train()
    for batch_idx, data in enumerate(progress_bar(train_loader, parent=mb)):
        data = data.to(device)
        optimizer.zero_grad()

        output = model(data)
        raise NotImplementedError
        loss = F.nll_loss(output)

        loss.backward()
        optimizer.step()

        # Report
        n_iter = (epoch - 1) * len(train_loader) + batch_idx
        with torch.no_grad():
            writer.add_scalar('train/loss', loss.item(), n_iter)


def test(args: argparse.Namespace, model: nn.Module, device: torch.device, test_loader: DataLoader,
         epoch: int) -> None:
    model.eval()

    test_loss: float = 0
    correct: float = 0

    with torch.no_grad():
        for data in progress_bar(test_loader, parent=mb):
            data = data.to(device)

            output = model(data)
            raise NotImplementedError
            test_loss += F.nll_loss(output, reduction='sum').item()

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
    parser.add_argument("--seed", type=int, default=0, help="シード値")
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
    unique_words = ['*', '+', '-', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=']

    train_dataset = SimplifiedDataset("./datasets/train.csv", vocab=unique_words,
                                      transform=transforms.Lambda(lambda x: torch.eye(len(unique_words))[x]))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              collate_fn=train_dataset.collate, shuffle=True, **kwargs)

    test_dataset = SimplifiedDataset("./datasets/val.csv", vocab=unique_words,
                                     transform=transforms.Lambda(lambda x: torch.eye(len(unique_words))[x]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.test_batch_size, collate_fn=test_dataset.collate, shuffle=False, **kwargs)

    # model, optimizerの用意
    model = SimplifiedNet().to(device)
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
