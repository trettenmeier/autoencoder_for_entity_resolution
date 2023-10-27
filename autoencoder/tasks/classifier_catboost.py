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
from ..classifier.model_catboost import train_catboost_classifier
from ..classifier.evaluate_catbost import evaluate_catboost_classifier


logger = logging.getLogger(__name__)


class TrainAndEvaluateClassifierCatboostTask(LuigiBaseTask):
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

        model = train_catboost_classifier(X_train, y_train)
        max_f1 = evaluate_catboost_classifier(model, X_test, y_test)

        joblib.dump(max_f1, self.output().path)

    def output(self) -> luigi.LocalTarget:
        output_path = os.path.join(
            get_path_to_working_dir(),
            "07_evaluation_results")

        os.makedirs(output_path, exist_ok=True)

        output_path = os.path.join(
            output_path,
            f"results_{self.embedding}__{self.model_class}__catboost_classifier.joblib"
        )
        return luigi.LocalTarget(output_path)
