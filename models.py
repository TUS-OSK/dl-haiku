#!/usr/bin/env python3
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


class SampleNet(nn.Module):
    def __init__(self) -> None:
        super(SampleNet, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class SimplifiedNet(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int = 64, lstm_hidden_dim: int = 128,
                 lstm_num_layers: int = 4, cvae_latent_size: int = 64) -> None:
        super(SimplifiedNet, self).__init__()
        self.encoder = SimplifiedEncoder(num_embeddings, embedding_dim, lstm_hidden_dim, lstm_num_layers)
        self.cvae = SimplifiedCVAE(lstm_hidden_dim * lstm_num_layers * 2,
                                   num_embeddings, embedding_dim, cvae_latent_size)
        self.decoder = SimplifiedDecoder(num_embeddings, embedding_dim, lstm_hidden_dim, lstm_num_layers)
        self.lstm_num_layers = lstm_num_layers
        self.lstm_hidden_dim = lstm_hidden_dim

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        h, c = self.encoder(x)
        hc = torch.cat([h, c], dim=0)

        reconstructioned_hc, z, mean, log_var, prior_mean, prior_log_var = self.cvae(
            hc.transpose(0, 1).reshape(x.size(1), -1), condition)

        h, c = reconstructioned_hc.view(x.size(1), self.lstm_num_layers * 2, self.lstm_hidden_dim) \
            .transpose(0, 1).chunk(2, dim=0)
        x = self.decoder(torch.cat([torch.ones(1, x.size(1), dtype=torch.long), x[1:]], dim=0), h, c)
        return x, z, mean, log_var, prior_mean, prior_log_var

    def infer(self, condition: torch.Tensor) -> torch.Tensor:
        reconstructioned_hc = self.cvae.infer(condition)

        h, c = reconstructioned_hc.view(condition.size(0), self.lstm_num_layers * 2, self.lstm_hidden_dim) \
            .transpose(0, 1).chunk(2, dim=0)
        x = self.decoder.infer(h, c)
        return x


class SimplifiedEncoder(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, hidden_dim: int, num_layers: int) -> None:
        super(SimplifiedEncoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.lstm = [nn.LSTMCell(embedding_dim, hidden_dim)] + \
                    [nn.LSTMCell(hidden_dim, hidden_dim) for _ in range(num_layers - 1)]
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        embed = self.embedding(x)
        hiddens = self.init_hidden(x.size(1))

        for sentence_words, raw in zip(embed, x):
            for lstm, hidden in zip(self.lstm, hiddens):
                h, c = lstm(sentence_words, hidden)
                sentence_words = h
                hidden[0] = torch.where(raw == 0, hidden[0].transpose(0, 1), h.transpose(0, 1)).transpose(0, 1)
                hidden[1] = torch.where(raw == 0, hidden[1].transpose(0, 1), c.transpose(0, 1)).transpose(0, 1)

        h, c = zip(*hiddens)
        h = torch.stack(h)
        c = torch.stack(c)
        return h, c

    def init_hidden(self, batch: int) -> Tuple[torch.Tensor]:
        weight = next(self.parameters())
        return [[weight.new_zeros(batch, self.hidden_dim)] * 2] * self.num_layers


class SimplifiedDecoder(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, hidden_dim: int, num_layers: int) -> None:
        super(SimplifiedDecoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.lstm = [nn.LSTMCell(embedding_dim, hidden_dim)] + \
                    [nn.LSTMCell(hidden_dim, hidden_dim) for _ in range(num_layers - 1)]
        self.fc = nn.Linear(hidden_dim, num_embeddings)

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        hiddens = [list(hidden) for hidden in zip(h.unbind(0), c.unbind(0))]
        outputs = []

        for sentence_words in x:
            for lstm, hidden in zip(self.lstm, hiddens):
                h, c = lstm(sentence_words, hidden)
                sentence_words = h
                hidden[0] = h
                hidden[1] = c
            outputs.append(h)

        x = torch.stack(outputs)
        x = self.fc(x)
        return x

    def infer(self, h: torch.Tensor, c: torch.Tensor, max_length: int = 20) -> torch.Tensor:
        hiddens = [list(hidden) for hidden in zip(h.unbind(0), c.unbind(0))]
        outputs = []

        sentence_words = torch.zeros(h.size(1), self.embedding_dim)

        for _ in range(max_length):
            for lstm, hidden in zip(self.lstm, hiddens):
                h, c = lstm(sentence_words, hidden)
                sentence_words = h
                hidden[0] = h
                hidden[1] = c
            output = self.fc(h)
            output = torch.cat([output[:, 1:2], output[:, 3:]], dim=1)
            output = F.softmax(output, dim=1)
            output = output.argmax(1)
            output = torch.where(output == 0, torch.ones_like(output), output + 2)
            outputs.append(output)
            sentence_words = self.embedding(output)
        output = torch.stack(outputs, 1)
        return output


class SimplifiedCVAE(nn.Module):
    def __init__(self, input_output_size: int, num_embeddings: int, condition_size: int, latent_size: int) -> None:
        super(SimplifiedCVAE, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, condition_size, padding_idx=0)
        self.encoder = nn.Sequential(
            nn.Linear(input_output_size + condition_size, input_output_size + condition_size),
            nn.ReLU(),
            nn.Linear(input_output_size + condition_size, input_output_size + condition_size),
            nn.ReLU(),
            nn.Linear(input_output_size + condition_size, input_output_size + condition_size),
            nn.ReLU(),
            nn.Linear(input_output_size + condition_size, latent_size * 2)
        )
        self.prior = nn.Sequential(
            nn.Linear(condition_size, latent_size * 2),
            nn.ReLU(),
            nn.Linear(latent_size * 2, latent_size * 2),
            nn.ReLU(),
            nn.Linear(latent_size * 2, latent_size * 2),
            nn.ReLU(),
            nn.Linear(latent_size * 2, latent_size * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size + condition_size, input_output_size),
            nn.ReLU(),
            nn.Linear(input_output_size, input_output_size),
            nn.ReLU(),
            nn.Linear(input_output_size, input_output_size),
            nn.ReLU(),
            nn.Linear(input_output_size, input_output_size)
        )

    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mean + eps * std

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor]:
        condition = self.embedding(condition)

        mean, log_var = self.encoder(torch.cat([x, condition], dim=1)).chunk(2, dim=1)
        prior_mean, prior_log_var = self.prior(condition).chunk(2, dim=1)

        z = self.reparameterize(mean, log_var)
        x = self.decoder(torch.cat([z, condition], dim=1))
        return x, z, mean, log_var, prior_mean, prior_log_var

    def infer(self, condition: torch.Tensor) -> torch.Tensor:
        condition = self.embedding(condition)

        prior_mean, prior_log_var = self.prior(condition).chunk(2, dim=1)

        z = self.reparameterize(prior_mean, prior_log_var)
        x = self.decoder(torch.cat([z, condition], dim=1))
        return x
