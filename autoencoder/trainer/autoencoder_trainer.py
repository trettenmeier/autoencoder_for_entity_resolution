import os
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow

from ..utils.get_resources import get_config, get_path_to_working_dir


logger = logging.getLogger(__name__)


def autoencoder_trainer(model, train_loader, val_loader):
    config = get_config()

    trainer = AutoencoderTrainer(
        model=model,
        num_epochs=config["autoencoder_trainer"]["num_epochs"],
        learning_rate=config["autoencoder_trainer"]["learning_rate"],
        train_loader=train_loader,
        val_loader=val_loader,
        model_path=get_path_to_working_dir()
    )

    trainer.train()
    return trainer.load_best_model()


class AutoencoderTrainer:
    def __init__(self, model, train_loader, val_loader, num_epochs, learning_rate, model_path):
        self.device = None

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.model = model
        self.model_path = model_path
        os.makedirs(model_path, exist_ok=True)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.num_epochs = num_epochs

        self.min_valid_loss = np.inf

        mlflow.log_params({
            "autoencoder_num_epochs": num_epochs,
            "autoencoder_learning_rate": learning_rate,
            "autoencoder_loss": self.criterion,
            "autoencoder_optimizer": self.optimizer,
            "autoencoder_number_of_trainable_parameter": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        })

        try:
            mlflow.log_text(str(self.model), "model.txt")
        except:
            logger.warning("Can't model architecture to disk")

    def train(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        logger.info(self.model)
        self.model.to(self.device)

        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0.0
            for input_a, input_b in self.train_loader:
                train_loss += self.train_on_batch(input_a, input_b)

            valid_loss = self.validate()

            # output to console
            avg_train_loss = train_loss / len(self.train_loader)
            avg_val_loss = valid_loss / len(self.val_loader)
            logger.info(f"Epoch: {epoch + 1}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
            mlflow.log_metrics({
                "autoencoder_train_loss": avg_train_loss,
                "autoencoder_val_loss": avg_val_loss
            }, epoch)

            self.save_best_model(valid_loss)

    def train_on_batch(self, input_a, input_b):
        # randomly change the input and the goal
        ref_model_input = random.choice([input_a, input_b])
        ref_expected = random.choice([input_a, input_b])

        # clone them, so they won't reference the same object in memory
        model_input = ref_model_input.detach().clone()
        expected = ref_expected.detach().clone()

        model_input, expected = self.move_to_device(model_input, expected)
        expected = self.prepare_input(expected)

        reconstructed = self.model(model_input)
        loss = self.criterion(reconstructed, expected)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save_best_model(self, valid_loss):
        if self.min_valid_loss > valid_loss:
            self.min_valid_loss = valid_loss
            file_location = os.path.join(
                self.model_path,
                'temp_file_autoencoder_best_val_loss.pth'
            )
            torch.save(self.model.state_dict(), file_location)

    def load_best_model(self):
        file_location = os.path.join(
            self.model_path,
            "temp_file_autoencoder_best_val_loss.pth"
        )
        self.model.load_state_dict(
            torch.load(file_location, map_location=torch.device(self.device))
        )
        return self.model

    def move_to_device(self, a, b):
        a = a.to(self.device)
        b = b.to(self.device)
        return a, b

    def prepare_input(self, a):
        if hasattr(self.model, "reshape_input") and callable(self.model.reshape_input):
            a = self.model.reshape_input(a)
        return a

    def validate(self):
        valid_loss = 0.0

        self.model.eval()
        for input_a, input_b in self.val_loader:
            input_a, input_b = self.move_to_device(input_a, input_b)

            output_a = self.model(input_a)
            output_b = self.model(input_b)

            input_a = self.prepare_input(input_a)
            input_b = self.prepare_input(input_b)

            loss_a = self.criterion(output_a, input_a)
            loss_b = self.criterion(output_b, input_b)
            valid_loss += loss_a.item() * input_a.size(0)
            valid_loss += loss_b.item() * input_b.size(0)
        valid_loss = valid_loss / 2  # it's the loss from both datapoints, divide by 2

        return valid_loss
