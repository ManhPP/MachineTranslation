from abc import ABC

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import *


class Encoder(nn.Module, ABC):
    def __init__(self, vocab_size, embedding_dim, encoder_units, batch_size):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.encoder_units = encoder_units
        self.batch_size = batch_size

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=encoder_units)

    def forward(self, x, lengths, device):
        """

        :param x: (batch_size, max_length)
        :param lengths:
        :param device:
        :return:
        """

        x = self.embedding(x)
        # x: (batch_size, max_length, embedding_dim)

        x = pack_padded_sequence(x, lengths=lengths)
        hidden = self.init_hidden(device)
        output, hidden = self.gru(x, hidden)
        # output: (max_lengths, batch_size, encoder_units)
        # self.hidden: (1, batch_size, encoder_units)

        return output, hidden

    def init_hidden(self, device):
        return torch.zeros((1, self.batch_size, self.encoder_units), device=device)


class Decoder(nn.Module, ABC):
    def __init__(self, vocab_size, embedding_dim, decoder_units, encoder_units, batch_size):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.decoder_units = decoder_units
        self.encoder_units = encoder_units
        self.batch_size = batch_size

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.gru = nn.GRU(input_size=embedding_dim + encoder_units, hidden_size=decoder_units, batch_first=True)
        self.fc = nn.Linear(in_features=encoder_units, out_features=vocab_size)

        self.W1 = nn.Linear(encoder_units, decoder_units)
        self.W2 = nn.Linear(encoder_units, decoder_units)
        self.V = nn.Linear(encoder_units, 1)

    def forward(self, x, hidden, encoder_output):
        """

        :param x:
        :param hidden: (1, batch_size, encoder_units)
        :param encoder_output: (max_length, batch_size, encoder_units)
        :return:
        """
        encoder_output = encoder_output.permute(1, 0, 2)
        # encoder_output: (batch_size, max_length, encoder_units)

        hidden = hidden.permute(1, 0, 2)
        # hidden: (batch_size, 1, encoder_units)

        score = torch.tanh(self.W1(encoder_output) + self.W2(hidden))
        # score: (batch_size, max_length, encoder_units)

        attention_weights = F.softmax(self.V(score), dim=1)
        # attention_weights: (batch_size, max_length, encoder_units)

        context_vector = attention_weights * encoder_output
        context_vector = torch.sum(context_vector, dim=1)
        # context_vector: (batch_size, encoder_units)

        x = self.embedding(x)
        # x: (batch_size, 1, embedding_dim)

        x = torch.cat((context_vector.unsqueeze(1), x), -1)
        # x: (batch_size, 1, encoder_units + embedding_dims)

        output, state = self.gru(x)
        # output:
        # state:

        output = output.view(-1, output.size[2])
        x = self.fc(output)

        return x, state, attention_weights

    def init_hidden(self, device):
        return torch.zeros((1, self.batch_size, self.decoder_units), device=device)
