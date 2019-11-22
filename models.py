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
                 lstm_num_layers: int = 1, cvae_latent_size: int = 2) -> None:
        super(SimplifiedNet, self).__init__()
        assert lstm_num_layers == 1
        self.encoder = SimplifiedEncoder(num_embeddings, embedding_dim, lstm_hidden_dim, lstm_num_layers)
        self.cvae = SimplifiedCVAE(lstm_hidden_dim * lstm_num_layers * 2,
                                   num_embeddings, embedding_dim, cvae_latent_size)
        self.decoder = SimplifiedDecoder(num_embeddings, embedding_dim, lstm_hidden_dim, lstm_num_layers)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        h, c = self.encoder(x)
        hc = torch.cat([h, c], dim=1)

        reconstructioned_hc, z, mean, log_var, prior_mean, prior_log_var = self.cvae(hc, condition)

        h, c = reconstructioned_hc.chunk(2, dim=1)
        x = self.decoder(torch.cat([torch.ones(1, x.size(1), dtype=torch.long), x[1:]],
                                   dim=0), h.unsqueeze(0), c.unsqueeze(0))
        return x, z, mean, log_var, prior_mean, prior_log_var


class SimplifiedEncoder(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, hidden_dim: int, num_layers: int) -> None:
        super(SimplifiedEncoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTMCell(embedding_dim, hidden_dim)
        self.h0 = torch.randn(hidden_dim)
        self.c0 = torch.randn(hidden_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        embed = self.embedding(x)
        h = self.h0.expand(x.size(1), -1)
        c = self.c0.expand(x.size(1), -1)

        for sentence_words, raw in zip(embed, x):
            pre_h = h
            pre_c = c
            h, c = self.lstm(sentence_words, (h, c))
            h = torch.where(raw == 0, pre_h.transpose(0, 1), h.transpose(0, 1)).transpose(0, 1)
            c = torch.where(raw == 0, pre_c.transpose(0, 1), c.transpose(0, 1)).transpose(0, 1)
        return h, c


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
    def __init__(self, input_output_size: int, num_embeddings: int, condition_size: int, latent_size: int) -> None:
        super(SimplifiedCVAE, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, condition_size, padding_idx=0)
        self.encoder = nn.Sequential(
            nn.Linear(input_output_size + condition_size, input_output_size + condition_size),
            nn.ReLU(),
            nn.Linear(input_output_size + condition_size, latent_size*2)
        )
        self.prior = nn.Sequential(
            nn.Linear(condition_size, latent_size*2),
            nn.ReLU(),
            nn.Linear(latent_size*2, latent_size*2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size + condition_size, input_output_size),
            nn.ReLU(),
            nn.Linear(input_output_size, input_output_size)
        )

    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)

        return mean + eps*std

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor]:
        condition = self.embedding(condition)

        mean, log_var = self.encoder(torch.cat([x, condition], dim=1)).chunk(2, dim=1)
        prior_mean, prior_log_var = self.prior(condition).chunk(2, dim=1)

        z = self.reparameterize(mean, log_var)
        x = self.decoder(torch.cat([z, condition], dim=1))
        return x, z, mean, log_var, prior_mean, prior_log_var
