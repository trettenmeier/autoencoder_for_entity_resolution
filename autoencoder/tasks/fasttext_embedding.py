import os

import joblib
import luigi
import pandas as pd

from .base_task import LuigiBaseTask
from ..utils.get_resources import get_config, get_path_to_input_data

from .preprocessing import PreprocessingTask
from ..tokenizer_and_word_embeddings.make_fasttext_embeddings import make_fasttext_embeddings


class FasttextEmbeddingTask(LuigiBaseTask):
    config = get_config()

    def requires(self):
        return {
            "PreprocessingTask": PreprocessingTask(),
        }

    def run(self):
        X_train = pd.read_hdf(self.input()["PreprocessingTask"].path, "X_train")

        fasttext_model = make_fasttext_embeddings(
            df=X_train,
            size_embedding_vector=self.config["fasttext_embedding"]["size_embedding_vector"],
            training_epochs=self.config["fasttext_embedding"]["training_epochs"],
            skip_gram=self.config["fasttext_embedding"]["skip_gram"],
            window=self.config["fasttext_embedding"]["window"]
        )

        joblib.dump(fasttext_model, self.output().path)

    def output(self) -> luigi.LocalTarget:
        output_path = os.path.join(get_path_to_input_data())

        os.makedirs(output_path, exist_ok=True)

        output_path = os.path.join(
            output_path,
            f"fasttext_model_embedding_vec_len_{self.config['fasttext_embedding']['size_embedding_vector']}.joblib")
        return luigi.LocalTarget(output_path)
