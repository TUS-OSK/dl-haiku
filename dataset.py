#!/usr/bin/env python3
from torch.utils.data import Dataset


class SampleDataset(Dataset):
    def __init__(self, root, train: bool = True, transforms=None) -> None:
        self.root = root
        self.train = train
        self.transforms = transforms

    def __getitem__(self, index: int):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError
