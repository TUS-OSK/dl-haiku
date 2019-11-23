#!/usr/bin/env python3
import argparse
import json
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from dataset import HaikuDataset
from models import SimplifiedNet


def infer(args: argparse.Namespace, model: nn.Module, device: torch.device, condition: torch.Tensor,
          vocab: Dataset, temp: float) -> str:
    model.eval()

    with torch.no_grad():
        condition = condition.to(device)
        output = model.infer(condition, temp)[0]
        output = [vocab.reverse_dict[word] for word in output.tolist()]
        try:
            return output[:output.index("[EOS]")]
        except ValueError:
            return output


def main() -> None:
    # Training settings
    parser = argparse.ArgumentParser(description='dl-haiku')
    parser.add_argument('--model', type=str, default="model_1.pt", help='保存したモデル')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='CUDAを用いない')
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

    dataset = HaikuDataset(args.root, train=True)
    dataset.analize_vocab(['[PAD]', '[EOS]', "[UNK]"] + dataset.vocab)

    # modelの用意
    model = SimplifiedNet(len(dataset.vocab)).to(device)
    model.load_state_dict(torch.load(args.model))

    # 推論
    for kigo in dataset.vocab_dict:
        data = torch.tensor([dataset.vocab_dict[kigo]])
        result = infer(args, model, device, data, dataset, 4)
        print(f"{kigo}: {''.join(result)}")


if __name__ == "__main__":
    main()
