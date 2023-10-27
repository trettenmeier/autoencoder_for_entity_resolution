import torch
import torch.nn as nn

from ..utils.get_resources import get_config


def create_simple_autoencoder_model():
    config = get_config()

    return Autoencoder(
        size_latent_space=config["simple_autoencoder"]["size_latent_space"],
        size_embedding_vector=config["embedding_layer"]["size_of_embeddings"],
        vocab_size=config["wordpiece_tokenizer"]["vocab_size"],
        number_of_tokens_per_datapoint=config["wordpiece_tokenizer"]["number_of_tokens_per_datapoint"],
    )


class Autoencoder(nn.Module):
    def __init__(self,  size_latent_space, size_embedding_vector, vocab_size, number_of_tokens_per_datapoint):
        super().__init__()

        self.in_features = number_of_tokens_per_datapoint * size_embedding_vector

        self.embedding_layer = nn.Embedding(
                    num_embeddings=vocab_size,
                    embedding_dim=size_embedding_vector,
                    padding_idx=0
                )

        self.encoder = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.BatchNorm1d(self.in_features),
            nn.Linear(in_features=self.in_features, out_features=1000),
            nn.ReLU(),
            nn.BatchNorm1d(1000),
            nn.Linear(in_features=1000, out_features=500),
            nn.ReLU(),
            nn.BatchNorm1d(500),
            nn.Linear(in_features=500, out_features=250),
            nn.ReLU(),
            nn.BatchNorm1d(250),
            nn.Linear(in_features=250, out_features=120),
            nn.ReLU(),
            nn.BatchNorm1d(120),
            nn.Linear(in_features=120, out_features=size_latent_space)
            )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=size_latent_space, out_features=120),
            nn.ReLU(),
            nn.BatchNorm1d(120),
            nn.Linear(in_features=120, out_features=250),
            nn.ReLU(),
            nn.BatchNorm1d(250),
            nn.Linear(in_features=250, out_features=500),
            nn.ReLU(),
            nn.BatchNorm1d(500),
            nn.Linear(in_features=500, out_features=1000),
            nn.ReLU(),
            nn.BatchNorm1d(1000),
            nn.Linear(in_features=1000, out_features=self.in_features),
            nn.BatchNorm1d(self.in_features)
        )

    def forward(self, x):
        x = self.embedding_layer(x)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def reshape_input(self, x):
        """
        This makes the input the same shape as if it was fed to the forward()-method. Is used for comparing original
        input to model output.
        """
        with torch.no_grad():
            x = self.embedding_layer(x)
            return x.flatten(start_dim=1)
