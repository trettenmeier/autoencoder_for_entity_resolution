import os

import joblib
import luigi
import pandas as pd

from .base_task import LuigiBaseTask
from ..utils.get_resources import get_config, get_path_to_working_dir

from .preprocessing import PreprocessingTask
from ..tokenizer_and_word_embeddings.wordpiece_tokenizer import wordpiece_tokenizer


class WordpieceTokenizerTask(LuigiBaseTask):
    """
    Luigi task to load data. Has no required tasks.
    Call:
    "luigi --module sonar_ai.tasks.load_data_task LoadDataTask
     --name-of-dataset-file sample_600v3.csv --local-scheduler --workers 1"
    from shell to load the data and generate the corresponding dataframe.
    """
    config = get_config()

    def requires(self):
        return {
            "PreprocessingTask": PreprocessingTask(),
        }

    def run(self):
        X_train = pd.read_hdf(self.input()["PreprocessingTask"].path, "X_train")
        tokenizer = wordpiece_tokenizer(X_train)

        joblib.dump(tokenizer, self.output().path)

    def output(self) -> luigi.LocalTarget:
        output_path = os.path.join(
            get_path_to_working_dir(),
            "03_tokenizer_and_embeddings")

        os.makedirs(output_path, exist_ok=True)

        output_path = os.path.join(
            output_path,
            "wordpiece_tokenizer.joblib"
        )
        return luigi.LocalTarget(output_path)
