import mlflow

from catboost import CatBoostClassifier


def train_catboost_classifier(X_train, y_train):
    depth = 10
    iterations = 1000
    learning_rate = 0.1
    l2_leaf_reg = 10

    mlflow.log_params({
        "classifier_type": "catboost",
        "classifier_catboost_depth": depth,
        "classifier_catboost_iterations": iterations,
        "classifier_catboost_learning_rate": learning_rate,
        "classifier_catboost_l2_leaf_reg": l2_leaf_reg,
    })

    model = CatBoostClassifier(
        depth=depth,
        iterations=iterations,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        allow_writing_files=False,
        silent=False,
    )
    model.fit(X_train, y_train)
    return model
