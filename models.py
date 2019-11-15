#!/usr/bin/env python3
from typing import Tuple

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
    def __init__(self, num_embeddings: int, embedding_dim: int = 3, lstm_hidden_dim: int = 32,
                 lstm_num_layers: int = 2, cvae_latent_size: int = 4) -> None:
        super(SimplifiedNet, self).__init__()
        self.encoder = SimplifiedEncoder(num_embeddings, embedding_dim, lstm_hidden_dim, lstm_num_layers)
        self.cvae = SimplifiedCVAE(lstm_hidden_dim * lstm_num_layers * 2, cvae_latent_size)
        self.decoder = SimplifiedDecoder(num_embeddings, embedding_dim, lstm_hidden_dim, lstm_num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, c = self.encoder(x)
        reconstructioned_hc, z = self.cvae(torch.cat([h, c], dim=0).permute(1, 0, 2).reshape(x.size(1), -1))
        h, c = reconstructioned_hc.view(reconstructioned_hc.size(0), h.size(0) +
                                        c.size(0), -1).permute(1, 0, 2).chunk(2, dim=0)
        x = self.decoder(torch.cat([torch.ones(1, x.size(1), dtype=torch.long), x[1:]], dim=0), h, c)
        return x


class SimplifiedEncoder(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, hidden_dim: int, num_layers: int) -> None:
        super(SimplifiedEncoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers)
        self.h0 = torch.randn(num_layers, 1, hidden_dim)
        self.c0 = torch.randn(num_layers, 1, hidden_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        x = self.embedding(x)
        _, hidden = self.lstm(x, (self.h0.expand(-1, x.size(1), -1), self.c0.expand(-1, x.size(1), -1)))
        return hidden


class SimplifiedDecoder(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, hidden_dim: int, num_layers: int) -> None:
        super(SimplifiedDecoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, num_embeddings)

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x, _ = self.lstm(x, (h, c))
        x = self.fc(x)
        return x


class SimplifiedCVAE(nn.Module):
    def __init__(self, input_output_size: int, latent_size: int) -> None:
        super(SimplifiedCVAE, self).__init__()
        self.fc1 = nn.Linear(input_output_size, latent_size)
        self.fc2 = nn.Linear(latent_size, input_output_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        z = self.fc1(x)
        x = self.fc2(z)
        return x, z
