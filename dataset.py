#!/usr/bin/env python3
import csv
from typing import Dict, List, Tuple, Set

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm

from pathlib import Path


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
    def __init__(self, data_file: str, vocab=None, transform=None, target_transform=None) -> None:
        self.data_file = data_file
        self.transform = transform
        self.target_transform = target_transform

        self.data = self.load_data(data_file)
        self.analize_vocab(vocab)

    def analize_vocab(self, vocab=None):
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

    def __getitem__(self, index: int) -> Tuple[torch.Tensor]:
        sentence = self.data[index] + [self.vocab[1]]  # <EOS>
        condition, sentence = sentence[0], sentence[1:]
        sentence = torch.tensor([self.vocab_dict[word] for word in sentence], dtype=torch.long)
        condition = torch.tensor([self.vocab_dict[word] for word in condition], dtype=torch.long)

        if self.transform is not None:
            sentence = self.transform(sentence)

        if self.target_transform is not None:
            condition = self.target_transform(condition)

        return sentence, condition

    @staticmethod
    def collate(batch: List[Tuple[torch.Tensor]]) -> torch.Tensor:
        sentence, condition = zip(*batch)
        sentence = pad_sequence(sentence)
        condition = torch.cat(condition)
        return sentence, condition


class HaikuDataset(Dataset):
    def __init__(self, root: str, vocab=None, transform=None, target_transform=None, vocab_num: int = 2000,
                 train: bool = True) -> None:
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.vocab_num = vocab_num
        self.train = train

        self.kigo = self.load_kigo(self.root)
        self.data = self.load_data(self.root, self.kigo, self.train)
        self.analize_vocab(vocab)

    def analize_vocab(self, vocab=None):
        self.vocab = vocab or self.get_vocab()
        self.vocab_dict = {word: i for i, word in enumerate(self.vocab)}
        self.reverse_dict = {i: word for i, word in enumerate(self.vocab)}

    @staticmethod
    def load_data(root: Path, kigo: Set[str], train: bool) -> List[Tuple[str, List[str]]]:
        if train:
            file = root / "dataset/haiku/haiku_wakati"
        else:
            file = root / "dataset/haiku/haiku_wakati_test"
        with open(str(file)) as f:
            haiku = [sentence.split(" ")[:-1] for sentence in f.read().split("\n")]

        data = []
        for sentence in haiku:
            is_kigo = [word in kigo for word in sentence]
            if sum(is_kigo) == 0:
                continue
            elif sum(is_kigo) > 1:
                kigos = {s for s, is_k in zip(sentence, is_kigo) if is_k}
                if len(kigos) == 1:
                    continue
            data.append((sentence[is_kigo.index(True)], sentence))
        return data

    @staticmethod
    def load_kigo(root: str) -> Set[str]:
        kigo_files = root / "dataset/kigo"
        kigos = []
        for file in kigo_files.glob("*"):
            with open(file) as f:
                data = f.read().split()
                kigos.extend(data)
        return set(kigos)

    def get_vocab(self) -> List[str]:
        vocab = {}
        for sentence in tqdm(self.data):
            for word in sentence[1]:
                vocab[word] = vocab.get(word, 0) + 1

        vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
        return [v[0] for v in vocab[:self.vocab_num]]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor]:
        condition, sentence = self.data[index]
        sentence = sentence + [self.vocab[1]]
        sentence = torch.tensor([self.vocab_dict.get(word, 2) for word in sentence], dtype=torch.long)
        condition = torch.tensor(self.vocab_dict.get(condition, 2), dtype=torch.long)

        if self.transform is not None:
            sentence = self.transform(sentence)

        if self.target_transform is not None:
            condition = self.target_transform(condition)

        return sentence, condition

    @staticmethod
    def collate(batch: List[Tuple[torch.Tensor]]) -> torch.Tensor:
        sentence, condition = zip(*batch)
        sentence = pad_sequence(sentence)
        condition = torch.cat(condition)
        return sentence, condition


if __name__ == "__main__":
    dataset = HaikuDataset("./datasets/")
    print(len(dataset))
    dataset.analize_vocab(['[PAD]', '[EOS]', "[UNK]"] + dataset.vocab)
    print(f"dataset[0]: {dataset[0]}")
    print(f"dataset[-1]: {dataset[-1]}")
