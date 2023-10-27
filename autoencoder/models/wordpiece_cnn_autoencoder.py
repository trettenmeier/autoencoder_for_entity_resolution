import torch.nn as nn

from ..utils.get_resources import get_config


def create_cnn_autoencoder_model():
    config = get_config()

    return Autoencoder(
        size_latent_space=config["cnn_autoencoder"]["size_latent_space"],
        vocab_size=config["wordpiece_tokenizer"]["vocab_size"],
        number_of_tokens_per_datapoint=config["wordpiece_tokenizer"]["number_of_tokens_per_datapoint"],
        size_of_embeddings=config["embedding_layer"]["size_of_embeddings"]
    )


class Autoencoder(nn.Module):
    def __init__(self, size_latent_space, vocab_size, number_of_tokens_per_datapoint, size_of_embeddings):
        super().__init__()

        self.embedding_layer = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=size_of_embeddings,
            padding_idx=0
        )

        self.encoder = nn.Sequential(
            ## example code: self.conv_1 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_1, self.stride)
            # conv1d: (in_channels, out_channels, kernel_size, stride, padding)
            # in_channels: number of channels in the input image
            # out_channels: number of channels produced by the convolution
            # kernel size (int or tuple)
            # padding: (int, tuple or string): padding added to both sides of the input. string can be 'same' to keep the shape of the input
            # dilation is 1 per default!
            # kernel size 3 and padding 1 keeps the embedding dimension constant.
            nn.BatchNorm1d(number_of_tokens_per_datapoint),
            nn.Conv1d(in_channels=number_of_tokens_per_datapoint, out_channels=100, padding=1, dilation=1, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Conv1d(in_channels=100, out_channels=80, padding=1, dilation=1, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(80),
            nn.Conv1d(in_channels=80, out_channels=40, padding=1, dilation=1, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(40),
            nn.Conv1d(in_channels=40, out_channels=20, padding=1, dilation=1, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(20),
            nn.Conv1d(in_channels=20, out_channels=1, padding=1, dilation=1, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
        )

        self.unflattened_size = (1, 10)
        assert size_latent_space == self.unflattened_size[0] * self.unflattened_size[1], \
            "Error, size of latent space in config does not fit the actual size!"

        self.decoder = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=self.unflattened_size),
            nn.ConvTranspose1d(in_channels=1, out_channels=20, padding=1, dilation=1, kernel_size=3, stride=2, output_padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(20),
            nn.ConvTranspose1d(in_channels=20, out_channels=40, padding=1, dilation=1, kernel_size=3, stride=2, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(40),
            nn.ConvTranspose1d(in_channels=40, out_channels=80, padding=1, dilation=1, kernel_size=3, stride=2, output_padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(80),
            nn.ConvTranspose1d(in_channels=80, out_channels=100, padding=1, dilation=1, kernel_size=3, stride=2, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.ConvTranspose1d(in_channels=100, out_channels=number_of_tokens_per_datapoint, padding=1, dilation=1, kernel_size=3, stride=2, output_padding=1),
            nn.BatchNorm1d(number_of_tokens_per_datapoint),
        )

    def forward(self, x):
        embedding = self.embedding_layer(x)
        encoded = self.encoder(embedding)
        decoded = self.decoder(encoded)
        return decoded

    def reshape_input(self, input_embedding):
        return self.embedding_layer(input_embedding)
