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
        self.encoder = SimplifiedEncoder(num_embeddings, embedding_dim, lstm_hidden_dim, lstm_num_layers)
        self.cvae = SimplifiedCVAE(lstm_hidden_dim * lstm_num_layers * 2,
                                   num_embeddings, embedding_dim, cvae_latent_size)
        self.decoder = SimplifiedDecoder(num_embeddings, embedding_dim, lstm_hidden_dim, lstm_num_layers)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        h, c = self.encoder(x)
        hc = torch.cat([h, c], dim=0).permute(1, 0, 2).reshape(x.size(1), -1)
        reconstructioned_hc, z, mean, log_var, prior_z = self.cvae(hc, condition)
        h, c = reconstructioned_hc.view(reconstructioned_hc.size(0), h.size(0) +
                                        c.size(0), -1).permute(1, 0, 2).chunk(2, dim=0)
        x = self.decoder(torch.cat([torch.ones(1, x.size(1), dtype=torch.long), x[1:]], dim=0), h, c)
        return x, hc, reconstructioned_hc, z, prior_z,


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
    def __init__(self, input_output_size: int, num_embeddings: int, condition_size: int, latent_size: int) -> None:
        super(SimplifiedCVAE, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, condition_size, padding_idx=0)
        self.fc_mean = nn.Sequential(
            nn.Linear(input_output_size + condition_size, input_output_size + condition_size),
            nn.ReLU(),
            nn.Linear(input_output_size + condition_size, latent_size))
        self.fc_logvar = nn.Sequential(
            nn.Linear(input_output_size + condition_size, input_output_size + condition_size),
            nn.ReLU(),
            nn.Linear(input_output_size + condition_size, latent_size),
            nn.ReLU()
            )

        self.fc_c = nn.Sequential(
            nn.Linear(condition_size, latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, latent_size))

        self.fc2 = nn.Sequential(
            nn.Linear(latent_size + condition_size, input_output_size),
            nn.ReLU(),
            nn.Linear(input_output_size, input_output_size))
        
        def reparameterize(self, mean, log_var):
            std = torch.exp(0.5*log_var)
            eps = torch.randn_like(std)
            
            return mean + eps*std

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor]: # x shape: [N, 16]
        condition = self.embedding(condition)  # shape: [N, condition_size]

        mean = self.fc_mean(torch.cat([x, condition], dim=1)) # shape: [N, latent_size]
        log_var = self.fc_logvar(torch.cat([x, condition], dim=1)) # shape: [N, latent_size]

        z = self.reparameterize(mean, log_var) # shape: [N, latent_size]

        prior_z = self.fc_c(condition) # バッチ？ごとの潜在変数zを出力（ラベルから潜在変数zを推論） shape: [N, latent_size] 
        
        x = self.fc2(torch.cat([z, condition], dim=1)) # shape: [N, input_output_size]
        return x, z, mean, log_var, prior_z