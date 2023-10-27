import os
import logging
import joblib
import luigi
import pandas as pd

from .base_task import LuigiBaseTask
from ..utils.get_resources import get_config, get_path_to_working_dir
from .preprocessing import PreprocessingTask
from .fasttext_embedding import FasttextEmbeddingTask
from ..pytorch_dataloader.mp_fasttext import create_pytorch_dataloader_train_test as dataloader_fasttext
from ..pytorch_dataloader.mp_wordpiece import create_pytorch_dataloader_train_test as dataloader_wordpiece
from ..classifier.feed_data_through_autoencoder import feed_data_through_trained_autoencoder
from ..tasks.train_autoencoder import TrainAutoencoderTask
from ..tasks.wordpiece_tokenizer import WordpieceTokenizerTask

logger = logging.getLogger(__name__)


class FeedDataThroughAutoencoderTask(LuigiBaseTask):
    """
    """
    embedding = luigi.Parameter()
    model_class = luigi.Parameter()

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

        # model type
        if self.model_class == "simple":
            requires["Model"] = TrainAutoencoderTask(model_class="simple", embedding=self.embedding)
        elif self.model_class == "cnn":
            requires["Model"] = TrainAutoencoderTask(model_class="cnn", embedding=self.embedding)
        elif self.model_class == "rnn":
            requires["Model"] = TrainAutoencoderTask(model_class="rnn", embedding=self.embedding)
        elif self.model_class == "vae_cnn":
            requires["Model"] = TrainAutoencoderTask(model_class="vae_cnn", embedding=self.embedding)

        else:
            raise ValueError("model_class not valid.")

        return requires

    def run(self):
        logger.info("Loading data, fasttext model and creating dataloader.")
        X_train = pd.read_hdf(self.input()["PreprocessingTask"].path, "X_train")
        X_test = pd.read_hdf(self.input()["PreprocessingTask"].path, "X_test")

        preprocessing_model = joblib.load(self.input()["InputPreprocessing"].path)

        if self.embedding == "fasttext":
            train_loader, test_loader = dataloader_fasttext(X_train, X_test, preprocessing_model)
        elif self.embedding == "wordpiece":
            train_loader, test_loader = dataloader_wordpiece(X_train, X_test, preprocessing_model)
        else:
            raise ValueError("embedding not valid")

        logger.info("Loading model.")
        model = joblib.load(self.input()["Model"].path)

        logger.info("Feeding data through model.")
        output_training_data, output_test_data = feed_data_through_trained_autoencoder(model, train_loader, test_loader)

        df_train = pd.DataFrame(output_training_data)
        df_test = pd.DataFrame(output_test_data)

        df_train.to_hdf(self.output().path, "X_train")
        df_test.to_hdf(self.output().path, "X_test")

    def output(self) -> luigi.LocalTarget:
        output_path = os.path.join(
            get_path_to_working_dir(),
            "05_input_for_classifier")

        os.makedirs(output_path, exist_ok=True)

        output_path = os.path.join(
            output_path,
            f"train_test{self.embedding}_{self.model_class}.h5"
        )
        return luigi.LocalTarget(output_path)
