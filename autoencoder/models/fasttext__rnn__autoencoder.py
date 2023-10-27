# code frame from: https://github.com/hellojinwoo/TorchCoder/blob/master/autoencoders/rae.py

import torch.nn as nn

from ..utils.get_resources import get_config


def create_rnn_autoencoder_model():
    config = get_config()

    return Autoencoder(
        number_of_tokens_per_datapoint=config["fasttext_embedding"]["truncate_vectors_to_x_words"],
        size_of_embeddings=config["fasttext_embedding"]["size_embedding_vector"],
        size_latent_space=config["rnn_autoencoder"]["size_latent_space"],
    )


class Autoencoder(nn.Module):
    def __init__(self, number_of_tokens_per_datapoint, size_latent_space, size_of_embeddings):
        super(Autoencoder, self).__init__()

        # self.embedding_layer = nn.Embedding(
        #     num_embeddings=vocab_size,
        #     embedding_dim=size_of_embeddings,
        #     padding_idx=0
        # )

        # self.BatchNorm1D = nn.BatchNorm1d(99)

        self.encoder = Encoder(
            number_of_tokens_per_datapoint=number_of_tokens_per_datapoint,
            size_of_embeddings=size_of_embeddings,
            size_latent_space=size_latent_space
        )
        self.decoder = Decoder(
            number_of_tokens_per_datapoint=number_of_tokens_per_datapoint,
            size_latent_space=size_latent_space,
            size_of_embeddings=size_of_embeddings
        )

    def forward(self, x):
        # x = self.BatchNorm1D(x)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def reshape_input(self, input_embedding):
        return input_embedding


class Encoder(nn.Module):
    def __init__(self, number_of_tokens_per_datapoint, size_of_embeddings, size_latent_space):
        super().__init__()

        self.number_of_tokens_per_datapoint = number_of_tokens_per_datapoint
        self.size_of_embeddings = size_of_embeddings
        self.size_latent_space = size_latent_space

        self.batch_norm = nn.BatchNorm1d(self.number_of_tokens_per_datapoint)

        self.LSTM1 = nn.LSTM(
            input_size=size_of_embeddings,
            hidden_size=self.size_latent_space*3,
            num_layers=1,
            batch_first=True
        )

        self.LSTM2 = nn.LSTM(
            input_size=self.size_latent_space*3,
            hidden_size=self.size_latent_space*2,
            num_layers=1,
            batch_first=True
        )

        self.LSTM_last = nn.LSTM(
            input_size=self.size_latent_space*2,
            hidden_size=size_latent_space,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        x = self.batch_norm(x)
        x, (_, _) = self.LSTM1(x)
        x, (_, _) = self.LSTM2(x)
        x, (hidden_state, cell_state) = self.LSTM_last(x)
        last_lstm_layer_hidden_state = hidden_state[-1, :, :]
        return last_lstm_layer_hidden_state


class Decoder(nn.Module):
    def __init__(self, number_of_tokens_per_datapoint, size_latent_space, size_of_embeddings):
        super().__init__()

        self.number_of_tokens_per_datapoint = number_of_tokens_per_datapoint
        self.size_latent_space = size_latent_space
        self.size_of_embeddings = size_of_embeddings

        self.LSTM1 = nn.LSTM(
            input_size=size_latent_space,
            hidden_size=size_latent_space*2,
            num_layers=1,
            batch_first=True
        )

        self.LSTM2 = nn.LSTM(
            input_size=size_latent_space*2,
            hidden_size=size_latent_space*3,
            num_layers=1,
            batch_first=True
        )

        self.LSTM_last = nn.LSTM(
            input_size=size_latent_space*3,
            hidden_size=self.size_of_embeddings,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(self.size_of_embeddings, self.size_of_embeddings)

        self.batch_norm = nn.BatchNorm1d(self.number_of_tokens_per_datapoint)

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, self.number_of_tokens_per_datapoint, 1)
        x, (_, _) = self.LSTM1(x)
        x, (_, _) = self.LSTM2(x)
        x, (_, _) = self.LSTM_last(x)
        x = x.reshape((-1, self.number_of_tokens_per_datapoint, self.size_of_embeddings))
        out = self.fc(x)
        out = self.batch_norm(out)
        return out

