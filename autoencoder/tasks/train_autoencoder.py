import os
import logging
import joblib
import luigi
import pandas as pd
import mlflow

from .base_task import LuigiBaseTask
from ..utils.get_resources import get_config, get_path_to_working_dir
from .preprocessing import PreprocessingTask
from .fasttext_embedding import FasttextEmbeddingTask
from ..pytorch_dataloader.mp_fasttext import create_pytorch_dataloader_fasttext_for_autoencoder_training
from ..models.fasttext__simple_autoencoder import create_simple_autoencoder_model as fasttext__simple_autoencoder
from ..models.fasttext__cnn_autoencoder import create_cnn_autoencoder_model as fasttext__cnn_autoencoder
from ..models.fasttext__rnn__autoencoder import create_rnn_autoencoder_model as fasttext__rnn_autoencoder
from ..trainer.autoencoder_trainer import autoencoder_trainer
from ..trainer.vae_trainer import vae_autoencoder_trainer
from .wordpiece_tokenizer import WordpieceTokenizerTask
from ..pytorch_dataloader.mp_wordpiece import create_pytorch_dataloader_wordpiece_for_autoencoder_training
from ..models.wordpiece_simple_autoencoder import create_simple_autoencoder_model as wordpiece__simple_autoencoder
from ..models.wordpiece_cnn_autoencoder import create_cnn_autoencoder_model as wordpiece__cnn_autoencoder
from ..models.wordpiece_rnn_autoencoder import create_rnn_autoencoder_model as wordpiece__rnn_autoencoder

logger = logging.getLogger(__name__)


class TrainAutoencoderTask(LuigiBaseTask):
    """
    Instanciates DataLoader, Model and Trainer. Then trains the autoencoder-model and saves the one with the lowest
    validation loss.

    model_class = "simple", "cnn", "rnn", "vae_cnn"
    embedding = "fasttext", "wordpiece"
    """
    model_class = luigi.Parameter()
    embedding = luigi.Parameter()

    config = get_config()

    def requires(self):
        requires = {
            "PreprocessingTask": PreprocessingTask()
        }

        # input preprocessing
        if self.embedding == "fasttext":
            requires["InputPreprocessing"] = FasttextEmbeddingTask()
        elif self.embedding == "wordpiece":
            requires["InputPreprocessing"] = WordpieceTokenizerTask()
        else:
            raise ValueError("embedding not valid.")

        return requires

    def run(self):
        logger.info("Loading data.")
        X_train_only_matches = pd.read_hdf(self.input()["PreprocessingTask"].path, "X_train_only_matches")

        logger.info("Loading preprocessing model.")
        preprocessing_model = joblib.load(self.input()["InputPreprocessing"].path)
        if self.embedding == "fasttext":
            train_loader, val_loader = create_pytorch_dataloader_fasttext_for_autoencoder_training(
                X_train_only_matches, preprocessing_model)
            mlflow.log_params({
                "fasttext_embedding_size_embedding_vector": self.config["fasttext_embedding"]["size_embedding_vector"],
                "fasttext_embedding_training_epochs": self.config["fasttext_embedding"]["training_epochs"],
                "fasttext_embedding_skip_gram": self.config["fasttext_embedding"]["skip_gram"],
                "fasttext_embedding_window": self.config["fasttext_embedding"]["window"],
                "fasttext_embedding_truncate_vectors_to_x_words": self.config["fasttext_embedding"][
                    "truncate_vectors_to_x_words"],
            })
            logger.info("Creating model.")
            if self.model_class == "simple":
                model = fasttext__simple_autoencoder()
                mlflow.log_param("size_latent_space", self.config["simple_autoencoder"]["size_latent_space"])
            elif self.model_class == "cnn":
                model = fasttext__cnn_autoencoder()
                mlflow.log_param("size_latent_space", self.config["cnn_autoencoder"]["size_latent_space"])
            elif self.model_class == "rnn":
                model = fasttext__rnn_autoencoder()
                mlflow.log_param("size_latent_space", self.config["rnn_autoencoder"]["size_latent_space"])
            elif self.model_class == "vae_cnn":
                model = fasttext__vae_cnn_autoencoder()
                mlflow.log_param("size_latent_space", self.config["vae_cnn_autoencoder"]["size_latent_space"])

            else:
                raise ValueError("Invalid model class given.")

        elif self.embedding == "wordpiece":
            train_loader, val_loader = create_pytorch_dataloader_wordpiece_for_autoencoder_training(
                X_train_only_matches, preprocessing_model)
            mlflow.log_params({
                "wordpiece_tokenizer_vocab_size": self.config["wordpiece_tokenizer"]["vocab_size"],
                "wordpiece_tokenizer_number_of_tokens_per_datapoint": self.config["wordpiece_tokenizer"][
                    "number_of_tokens_per_datapoint"],
                "wordpiece_tokenizer_fixed_position_of_values": self.config["wordpiece_tokenizer"][
                    "fixed_position_of_values"],
                "wordpiece_size_per_embedding": self.config["embedding_layer"]["size_of_embeddings"]
            })
            logger.info("Creating model.")
            if self.model_class == "simple":
                model = wordpiece__simple_autoencoder()
                mlflow.log_param("size_latent_space", self.config["simple_autoencoder"]["size_latent_space"])
            elif self.model_class == "cnn":
                model = wordpiece__cnn_autoencoder()
                mlflow.log_param("size_latent_space", self.config["cnn_autoencoder"]["size_latent_space"])
            elif self.model_class == "rnn":
                model = wordpiece__rnn_autoencoder()
                mlflow.log_param("size_latent_space", self.config["rnn_autoencoder"]["size_latent_space"])
            else:
                raise ValueError("Invalid model class given.")

        else:
            raise ValueError("embedding not valid")

        logger.info("Training model.")
        mlflow.log_params({
            "model type": self.model_class,
            "embedding": self.embedding,
            "preprocessing_test_size": self.config["preprocessing"]["test_size"],
            "preprocessing_random_seed": self.config["preprocessing"]["random_seed"],
            "dataloader_shuffle": self.config["dataloader"]["shuffle"],
            "dataloader_random_seed": self.config["dataloader"]["random_seed"],
            "dataloader_batch_size": self.config["dataloader"]["batch_size"],
        })

        if "vae" in self.model_class:
            trained_model = vae_autoencoder_trainer(model, train_loader, val_loader)
        else:
            trained_model = autoencoder_trainer(model, train_loader, val_loader)

        joblib.dump(trained_model, self.output().path)

    def output(self) -> luigi.LocalTarget:
        output_path = os.path.join(
            get_path_to_working_dir(),
            "04_trained_autoencoders")

        os.makedirs(output_path, exist_ok=True)

        output_path = os.path.join(
            output_path,
            f"{self.embedding}__{self.model_class}_autoencoder.joblib"
        )
        return luigi.LocalTarget(output_path)
