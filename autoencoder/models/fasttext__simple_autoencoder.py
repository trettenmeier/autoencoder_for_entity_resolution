import torch.nn as nn
from ..utils.get_resources import get_config


def create_simple_autoencoder_model():
    config = get_config()

    return Autoencoder(
        size_latent_space=config["simple_autoencoder"]["size_latent_space"],
        truncate_vectors_to_x_words=config["fasttext_embedding"]["truncate_vectors_to_x_words"],
        size_embedding_vector=config["fasttext_embedding"]["size_embedding_vector"]
    )


class Autoencoder(nn.Module):
    def __init__(self, size_latent_space, truncate_vectors_to_x_words, size_embedding_vector):
        super().__init__()

        self.size_latent_space = size_latent_space
        self.in_features = truncate_vectors_to_x_words * size_embedding_vector

        self.encoder = nn.Sequential(
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
            nn.Linear(in_features=120, out_features=self.size_latent_space)
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.size_latent_space, out_features=120),
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
        x = x.reshape(-1, self.in_features)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def reshape_input(self, input_embedding):
        return input_embedding.reshape(-1, self.in_features)
