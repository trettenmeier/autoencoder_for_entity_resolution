import pandas as pd


def get_only_matches_from_x_train(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    X["label"] = y
    X = X[X["label"] == 1]
    X.reset_index(drop=True, inplace=True)
    X = X.drop(columns="label", axis=1)
    return X
