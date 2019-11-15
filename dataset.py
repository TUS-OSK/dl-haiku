#!/usr/bin/env python3
import csv
from typing import Dict, List, Tuple

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm
import torch


class SampleDataset(Dataset):
    def __init__(self, root, train: bool = True, transform=None) -> None:
        self.root = root
        self.train = train
        self.transform = transform

    def __getitem__(self, index: int):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError


class SimplifiedDataset(Dataset):
    def __init__(self, data_file: str, vocab=None, transform=None) -> None:
        self.data_file = data_file
        self.transform = transform

        self.data = self.load_data(data_file)
        self.vocab = vocab or self.get_vocab()
        self.vocab_dict = {word: i for i, word in enumerate(self.vocab)}
        self.reverse_dict = {i: word for i, word in enumerate(self.vocab)}

    @staticmethod
    def load_data(data_file: str) -> List[List[str]]:
        with open(data_file) as f:
            data = list(csv.reader(f))
        return data

    def get_vocab(self) -> List[str]:
        vocab = set()
        for sentence in tqdm(self.data):
            for word in sentence:
                vocab.add(word)

        return sorted(vocab)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> List[str]:
        sentence = self.data[index]
        sentence = [self.vocab_dict[word] for word in sentence]

        if self.transform is not None:
            sentence = self.transform(sentence)

        return sentence

    @staticmethod
    def collate(batch: List[torch.Tensor]) -> torch.Tensor:
        batch = pad_sequence(batch)
        return batch


if __name__ == "__main__":
    dataset = SimplifiedDataset("./datasets/train.csv")
    print(len(dataset))
    print(f"dataset[0]: {dataset[0]}")
    print(f"dataset[-1]: {dataset[-1]}")
