#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from torch import nn

from dataset import SimplifiedDataset


class SampleNet(nn.Module):
    def __init__(self) -> None:
        super(SampleNet, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class SimplifiedNet(nn.Module):
    def __init__(self) -> None:
        super(SimplifiedNet, self).__init__()
        self.fc = nn.Linear(30, 30)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
