#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from torch import nn


class SampleNet(nn.Module):
    def __init__(self) -> None:
        super(SampleNet, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
