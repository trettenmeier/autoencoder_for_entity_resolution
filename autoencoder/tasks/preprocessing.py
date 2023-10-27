import os
import luigi
import pandas as pd

from .base_task import LuigiBaseTask
from ..utils.get_resources import get_config, get_path_to_working_dir, get_path_to_input_data

from ..preprocessing.split_data import split_data
from ..preprocessing.clean_data import clean_data
from ..preprocessing.get_only_matches_from_x_train import get_only_matches_from_x_train


class PreprocessingTask(LuigiBaseTask):
    """
    Luigi task to load data. Has no required tasks.
    Call:
    "luigi --module sonar_ai.tasks.load_data_task LoadDataTask
     --name-of-dataset-file sample_600k.csv --local-scheduler --workers 1"
    from shell to load the data and generate the corresponding dataframe.
    """
    config = get_config()

    def run(self) -> None:
        input_path = os.path.join(get_path_to_input_data(), self.config["input_file"])

        df = pd.read_csv(input_path)

        X_train, X_test, y_train, y_test = split_data(
            df=df,
            test_size=self.config["preprocessing"]["test_size"],
            random_seed=self.config["preprocessing"]["random_seed"]
        )

        X_train = clean_data(X_train)
        X_test = clean_data(X_test)

        X_train_only_matches = get_only_matches_from_x_train(X_train, y_train)

        X_train.to_hdf(self.output().path, "X_train")
        X_test.to_hdf(self.output().path, "X_test")
        X_train_only_matches.to_hdf(self.output().path, "X_train_only_matches")
        y_train.to_hdf(self.output().path, "y_train")
        y_test.to_hdf(self.output().path, "y_test")

    def output(self) -> luigi.LocalTarget:
        output_path = os.path.join(get_path_to_input_data(), "preprocessed_data.h5")
        return luigi.LocalTarget(output_path)
