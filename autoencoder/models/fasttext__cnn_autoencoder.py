import torch.nn as nn

from ..utils.get_resources import get_config


def create_cnn_autoencoder_model():
    config = get_config()

    return Autoencoder(
        size_latent_space=config["cnn_autoencoder"]["size_latent_space"],
        truncate_vectors_to_x_words=config["fasttext_embedding"]["truncate_vectors_to_x_words"]
    )


class Autoencoder(nn.Module):
    def __init__(self, size_latent_space, truncate_vectors_to_x_words):
        super().__init__()

        self.encoder = nn.Sequential(
            # example code: self.conv_1 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_1, self.stride)
            # conv1d: (in_channels, out_channels, kernel_size, stride, padding)
            # in_channels: number of channels in the input image
            # out_channels: number of channels produced by the convolution
            # kernel size (int or tuple)
            # padding: (int, tuple or string): padding added to both sides of the input. string can be 'same' to keep the shape of the input
            # dilation is 1 per default!
            # kernel size 3 and padding 1 keeps the embedding dimension constant.
            nn.BatchNorm1d(truncate_vectors_to_x_words),
            nn.Conv1d(in_channels=truncate_vectors_to_x_words, out_channels=50, padding=1, dilation=1, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Conv1d(in_channels=50, out_channels=40, padding=1, dilation=1, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(40),
            nn.Conv1d(in_channels=40, out_channels=20, padding=1, dilation=1, kernel_size=3, stride=2),
            nn.ReLU(),
            # nn.BatchNorm1d(80),
            # nn.Conv1d(in_channels=80, out_channels=70, padding=1, dilation=1, kernel_size=3, stride=2),
            # nn.ReLU(),
            # nn.BatchNorm1d(70),
            # nn.Conv1d(in_channels=70, out_channels=60, padding=1, dilation=1, kernel_size=3, stride=2),
            # nn.ReLU(),
            nn.Flatten(start_dim=1),
        )

        self.unflattened_size = (20, 19)
        assert size_latent_space == self.unflattened_size[0] * self.unflattened_size[1],\
            "Error, size of latent space in config does not fit the actual size!"

        self.decoder = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=self.unflattened_size),
            # nn.ConvTranspose1d(in_channels=60, out_channels=70, padding=1, dilation=1, kernel_size=3, stride=2, output_padding=0),
            # nn.ReLU(),
            # nn.BatchNorm1d(70),
            # nn.ConvTranspose1d(in_channels=70, out_channels=80, padding=1, dilation=1, kernel_size=4, stride=2, output_padding=0),
            # nn.ReLU(),
            # nn.BatchNorm1d(80),
            nn.ConvTranspose1d(in_channels=20, out_channels=40, padding=1, dilation=1, kernel_size=3, stride=2, output_padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(40),
            nn.ConvTranspose1d(in_channels=40, out_channels=50, padding=0, dilation=1, kernel_size=3, stride=2, output_padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.ConvTranspose1d(in_channels=50, out_channels=truncate_vectors_to_x_words, padding=1, dilation=1, kernel_size=3, stride=2, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(truncate_vectors_to_x_words),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def reshape_input(self, input_embedding):
        return input_embedding
