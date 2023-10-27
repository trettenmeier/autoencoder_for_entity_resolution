import random
import logging
import torch

from .autoencoder_trainer import AutoencoderTrainer
from ..utils.get_resources import get_config, get_path_to_working_dir


logger = logging.getLogger(__name__)


def vae_autoencoder_trainer(model, train_loader, val_loader):
    config = get_config()

    trainer = VAETrainer(
        model=model,
        num_epochs=config["autoencoder_trainer"]["num_epochs"],
        learning_rate=config["autoencoder_trainer"]["learning_rate"],
        train_loader=train_loader,
        val_loader=val_loader,
        model_path=get_path_to_working_dir()
    )

    trainer.train()
    return trainer.load_best_model()


class VAETrainer(AutoencoderTrainer):
    def train_on_batch(self, input_a, input_b):
        # randomly change the input and the goal
        ref_model_input = random.choice([input_a, input_b])
        ref_expected = random.choice([input_a, input_b])

        # clone them, so they won't reference the same object in memory
        model_input = ref_model_input.detach().clone()
        expected = ref_expected.detach().clone()

        model_input, expected = self.move_to_device(model_input, expected)
        expected = self.prepare_input(expected)

        encoded, z_mean, z_log_var, reconstructed = self.model(model_input)
        kl_div = -0.5 * torch.sum(1 + z_log_var
                                  - z_mean ** 2
                                  - torch.exp(z_log_var),
                                  axis=1)

        kl_div = kl_div.mean()

        pixelwise = self.criterion(reconstructed, expected)

        reconstruction_term_weight = 1  # this is a hyperparameter
        loss = reconstruction_term_weight * pixelwise + kl_div

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def validate(self):
        valid_loss = 0.0

        self.model.eval()
        with torch.no_grad():
            for input_a, input_b in self.val_loader:
                input_a, input_b = self.move_to_device(input_a, input_b)

                def get_loss(data_in):

                    encoded, z_mean, z_log_var, reconstructed = self.model(data_in)
                    kl_div = -0.5 * torch.sum(1 + z_log_var
                                              - z_mean ** 2
                                              - torch.exp(z_log_var),
                                              axis=1)  # sum over latent dimension

                    kl_div = kl_div.mean()

                    expected = self.prepare_input(data_in)

                    pixelwise = self.criterion(reconstructed, expected)

                    reconstruction_term_weight = 1
                    loss = reconstruction_term_weight * pixelwise + kl_div
                    return loss

                loss_a = get_loss(input_a)
                loss_b = get_loss(input_b)

                valid_loss += loss_a.item() * input_a.size(0)
                valid_loss += loss_b.item() * input_b.size(0)
            valid_loss = valid_loss / 2  # it's the loss from both datapoints, divide by 2

            return valid_loss
