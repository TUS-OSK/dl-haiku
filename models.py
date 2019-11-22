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
    def __init__(self, num_embeddings: int, embedding_dim: int = 2, lstm_hidden_dim: int = 8,
                 lstm_num_layers: int = 2, cvae_latent_size: int = 2) -> None:
        super(SimplifiedNet, self).__init__()
        self.encoder = SimplifiedEncoder(num_embeddings, embedding_dim, lstm_hidden_dim, lstm_num_layers)
        self.cvae = SimplifiedCVAE(lstm_hidden_dim * lstm_num_layers * 2, cvae_latent_size)
        self.decoder = SimplifiedDecoder(num_embeddings, embedding_dim, lstm_hidden_dim, lstm_num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_2, c_2 = self.encoder(x)
        hc = torch.cat([h_2, c_2], dim = 1)
        reconstructioned_hc, z = self.cvae(hc)

        #ここまで修正しました（64*16のhc）

        h, c = reconstructioned_hc.chunk(2, dim=1)
        x = self.decoder(torch.cat([torch.ones(1, x.size(1), dtype=torch.long), x[1:]], dim=0), h.unsqueeze(0), c.unsqueeze(0))
        return x, hc, reconstructioned_hc, z


class SimplifiedEncoder(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, hidden_dim: int, num_layers: int) -> None:
        super(SimplifiedEncoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTMCell(embedding_dim, hidden_dim)
        self.h0 = torch.randn(hidden_dim)
        self.c0 = torch.randn(hidden_dim)
        self.h0_2 = torch.randn(hidden_dim)
        self.c0_2 = torch.randn(hidden_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        embed = self.embedding(x)
        h = self.h0.expand(x.size(1), -1)
        c = self.c0.expand(x.size(1), -1)
        h_2 = self.h0_2.expand(x.size(1), -1)
        c_2 = self.c0_2.expand(x.size(1), -1)
        #x = x.reshape(8, 64)
        for sentence_words, raw in zip(embed, x):
            pre_h = h # maskingのため
            pre_c = c # maskingのため
            h, c = self.lstm(sentence_words, (h, c))
            h_2, c_2 = self.lstm(h, (h_2, c_2))
            h = torch.where(raw == 0, pre_h.transpose(0, 1), h.transpose(0, 1)).transpose(0, 1) # masking
            c = torch.where(raw == 0, pre_c.transpose(0, 1), c.transpose(0, 1)).transpose(0, 1) # masking
            h_2 = torch.where(raw == 0, pre_h.transpose(0, 1), h_2.transpose(0, 1)).transpose(0, 1) # masking
            c_2 = torch.where(raw == 0, pre_c.transpose(0, 1), c_2.transpose(0, 1)).transpose(0, 1) # masking

        """_, hidden = self.lstm(x, (self.h0.expand(-1, x.size(1), -1), self.c0.expand(-1, x.size(1), -1)))
        return hidden"""
        return h, c, h_2, c_2

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
        self.fc_mean = nn.Sequential(
            nn.Linear(input_output_size, input_output_size),
            nn.ReLU(),
            nn.Linear(input_output_size, latent_size))
        self.fc_std = nn.Sequential(
            nn.Linear(input_output_size, input_output_size),
            nn.ReLU(),
            nn.Linear(input_output_size, latent_size))
        self.fc2 = nn.Sequential(
            nn.Linear(latent_size, input_output_size),
            nn.ReLU(),
            nn.Linear(input_output_size, input_output_size))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        mean = self.fc_mean(x)
        std = self.fc_std(x)
        z = mean + torch.randn_like(std) * std
        x = self.fc2(z)
        return x, z
