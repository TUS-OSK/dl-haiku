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

from dataset import HaikuDataset
from models import SimplifiedNet

mb: ConsoleMasterBar
writer: SummaryWriter


def train(args: argparse.Namespace, model: nn.Module, device: torch.device, train_loader: DataLoader,
          optimizer: Optimizer, epoch: int) -> None:
    model.train()
    for batch_idx, (data, condition) in enumerate(progress_bar(train_loader, parent=mb)):
        data = data.to(device)
        condition = condition.to(device)
        optimizer.zero_grad()

        output, z, mean, log_var, prior_mean, prior_log_var = model(data, condition)

        recontruct_loss = F.cross_entropy(output.view(-1, output.size(2)), data.view(-1), ignore_index=0)
        cvae_constraint_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1).mean()
        prior_loss = -0.5 * torch.sum(1 + log_var + prior_log_var - (mean - prior_mean).pow(
            2) * prior_log_var.exp() - log_var.exp() * prior_log_var.exp(), dim=1).mean()

        loss = recontruct_loss + cvae_constraint_loss + prior_loss
        loss.backward()
        optimizer.step()

        # Report
        n_iter = (epoch - 1) * len(train_loader) + batch_idx
        with torch.no_grad():
            pred = output.argmax(dim=2)
            correct = (pred == data).sum().item() / (data != 1).sum().item()

            writer.add_scalar('train/recontruct_loss', recontruct_loss.item(), n_iter)
            writer.add_scalar('train/cvae_constraint_loss', cvae_constraint_loss.item(), n_iter)
            writer.add_scalar('train/prior_loss', prior_loss.item(), n_iter)
            writer.add_scalar('train/loss', loss.item(), n_iter)
            writer.add_scalar("train/correct", correct, n_iter)
            writer.add_text('train/ground_truth',
                            "".join(train_loader.dataset.reverse_dict[word] for word in data[:, 0].tolist()), n_iter)
            writer.add_text('train/predict',
                            "".join(train_loader.dataset.reverse_dict[word] for word in pred[:, 0].tolist()), n_iter)


def test(args: argparse.Namespace, model: nn.Module, device: torch.device, test_loader: DataLoader, epoch: int) -> None:
    model.eval()

    recontruct_loss: float = 0
    cvae_constraint_loss: float = 0
    prior_loss: float = 0
    loss: float = 0
    correct: float = 0

    with torch.no_grad():
        for data, condition in progress_bar(test_loader, parent=mb):
            data = data.to(device)
            condition = condition.to(device)

            output, z, mean, log_var, prior_mean, prior_log_var = model(data, condition)

            recontruct_loss += F.cross_entropy(
                output.view(-1, output.size(2)), data.view(-1), ignore_index=0, reduction="sum")
            cvae_constraint_loss += -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
            prior_loss += -0.5 * torch.sum(1 + log_var + prior_log_var - (mean - prior_mean).pow(
                2) * prior_log_var.exp() - log_var.exp() * prior_log_var.exp())

            loss += recontruct_loss + cvae_constraint_loss

            pred = output.argmax(dim=2)
            correct += pred.eq(data).sum().item()

        # Report
        recontruct_loss /= len(test_loader.dataset)
        cvae_constraint_loss /= len(test_loader.dataset)
        prior_loss /= len(test_loader.dataset)
        loss /= len(test_loader.dataset)

        writer.add_scalar("val/recontruct_loss", recontruct_loss, epoch)
        writer.add_scalar("val/cvae_constraint_loss", cvae_constraint_loss, epoch)
        writer.add_scalar("val/prior_loss", prior_loss, epoch)
        writer.add_scalar("val/loss", loss, epoch)
        writer.add_scalar("val/correct", correct, epoch)
        writer.add_text('val/ground_truth',
                        "".join(test_loader.dataset.reverse_dict[word] for word in data[:, 0].tolist()), epoch)
        writer.add_text(
            'val/predict', "".join(test_loader.dataset.reverse_dict[word] for word in pred[:, 0].tolist()), epoch)


def main() -> None:
    # Training settings
    parser = argparse.ArgumentParser(description='dl-haiku')
    parser.add_argument('--batch-size', type=int, default=43, metavar='N',
                        help='学習時のバッチサイズ (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='テスト時のバッチサイズ (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='エポック数 (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='CUDAを用いない')
    parser.add_argument('--not-save-model', action='store_false', default=True,
                        help='Modelを保存しない')
    parser.add_argument("--save_model", action='store_true', help="modelを保存する")
    parser.add_argument("--seed", type=int, default=0, help="シード値")
    parser.add_argument("--root", type=str, default="datasets", help="datasetsのflolder")
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

    train_dataset = HaikuDataset(args.root, train=True)
    train_dataset.analize_vocab(['[PAD]', '[EOS]', "[UNK]"] + train_dataset.vocab)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              collate_fn=train_dataset.collate, shuffle=True, **kwargs)

    test_dataset = HaikuDataset(args.root, vocab=train_dataset.vocab, train=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.test_batch_size, collate_fn=test_dataset.collate, shuffle=False, **kwargs)

    # model, optimizerの用意
    model = SimplifiedNet(len(train_dataset.vocab)).to(device)
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
