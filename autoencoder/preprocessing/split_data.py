import random
from typing import Tuple

import pandas as pd


def split_data(df: pd.DataFrame, test_size, random_seed) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    This function splits a Pandas DataFrame into train and test DataFrames while keeping parts with the same ID
    together, so a part_id can either be in train or test. This prevents the model a.o. from cheating by looking at
    specific substrings and knowing if it is a match or not just by 'memorizing' that substring. Example: In the
    600k_v4 dataset there are 1475 datapoints with 'page_internal_info_name' ==
    "RS PRO Universalgelenk, 1-fach, Glatt, Ausgebohrter Sitz, None, Aussen ø 42mm, 127mm, Stahl"
    that are all non-matches. A model with enough parameters can just learn that string and predict these datapoints
    'correctly' without even seeing the client part.
    """
    part_ids = list(df["part_id"].unique())

    random.Random(random_seed).shuffle(part_ids)

    split_index = round(len(part_ids) * (1 - test_size))
    part_ids_for_train = part_ids[:split_index]
    part_ids_for_test = part_ids[split_index:]

    df_train = df[df["part_id"].isin(part_ids_for_train)].copy()
    df_test = df[df["part_id"].isin(part_ids_for_test)].copy()

    y_train = df_train["label"].reset_index(drop=True)
    X_train = df_train.drop("label", axis=1).reset_index(drop=True)

    y_test = df_test["label"].reset_index(drop=True)
    X_test = df_test.drop("label", axis=1).reset_index(drop=True)

    return X_train, X_test, y_train, y_test
