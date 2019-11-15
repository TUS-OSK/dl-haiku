#!/usr/bin/env python3
import csv
from typing import List

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


class SimplifiedDataset(Dataset):
    def __init__(self, data_file: str, transforms=None) -> None:
        self.data_file = data_file
        self.transforms = transforms

        self.data = self.load_data(data_file)

    def load_data(self, data_file: str) -> List[List[str]]:
        with open(data_file) as f:
            data = list(csv.reader(f))
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> List[str]:
        return self.data[index]


if __name__ == "__main__":
    dataset = SimplifiedDataset("./dataset.csv")
    print(len(dataset))
    print(f"dataset[0]: {dataset[0]}")
    print(f"dataset[-1]: {dataset[-1]}")
