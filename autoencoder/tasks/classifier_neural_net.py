import os
import logging
import joblib
import luigi
import pandas as pd
import mlflow

from .base_task import LuigiBaseTask
from ..utils.get_resources import get_config, get_path_to_working_dir
from .preprocessing import PreprocessingTask
from .feed_data_through_autoencoder import FeedDataThroughAutoencoderTask
from ..classifier.dataloader_neural_net_classifier import create_pytorch_dataloader_for_neural_net_classifier
from ..classifier.model_neural_net import neural_net_classifier_model
from ..trainer.classifier_trainer import neural_net_classifier_trainer
from ..classifier.evaluate_neural_net import evaluate_neural_net_classifier

logger = logging.getLogger(__name__)


class TrainAndEvaluateClassifierNeuralNetTask(LuigiBaseTask):
    """
    """
    embedding = luigi.Parameter()
    model_class = luigi.Parameter()

    config = get_config()

    def requires(self):
        return {
            "PreprocessingTask": PreprocessingTask(),
            "FeedDataThroughAutoencoderTask": FeedDataThroughAutoencoderTask(
                embedding=self.embedding, model_class=self.model_class
            )
        }

    def run(self):
        X_train = pd.read_hdf(self.input()["FeedDataThroughAutoencoderTask"].path, "X_train")
        X_test = pd.read_hdf(self.input()["FeedDataThroughAutoencoderTask"].path, "X_test")
        y_train = pd.read_hdf(self.input()["PreprocessingTask"].path, "y_train")
        y_test = pd.read_hdf(self.input()["PreprocessingTask"].path, "y_test")

        classifier_train_loader, classifier_val_loader, classifier_test_loader = \
            create_pytorch_dataloader_for_neural_net_classifier(X_train,
                                                                X_test,
                                                                y_train,
                                                                y_test)

        if self.model_class == "simple":
            size_latent = self.config["simple_autoencoder"]["size_latent_space"]

        elif self.model_class == "cnn":
            size_latent = self.config["cnn_autoencoder"]["size_latent_space"]

        elif self.model_class == "rnn":
            size_latent = self.config["rnn_autoencoder"]["size_latent_space"]

        else:
            raise ValueError("Invalid model class given.")

        model = neural_net_classifier_model(size_latent)
        mlflow.log_param("size_latent_space_autoencoder", size_latent)
        trained_model = neural_net_classifier_trainer(model, classifier_train_loader, classifier_val_loader)

        max_f1 = evaluate_neural_net_classifier(trained_model, classifier_test_loader, y_test)

        joblib.dump(max_f1, self.output().path)

    def output(self) -> luigi.LocalTarget:
        output_path = os.path.join(
            get_path_to_working_dir(),
            "07_evaluation_results")

        os.makedirs(output_path, exist_ok=True)

        output_path = os.path.join(
            output_path,
            f"results_{self.embedding}__{self.model_class}__neural_net_classifier.joblib"
        )
        return luigi.LocalTarget(output_path)
