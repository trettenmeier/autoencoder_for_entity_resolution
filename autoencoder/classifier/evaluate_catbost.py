import logging

from ..metrics.metricsbag import MetricsBag

logger = logging.getLogger(__name__)


def evaluate_catboost_classifier(model, X_test, y_test):
    probabilities = model.predict_proba(X_test)

    bag = MetricsBag(y_test, probabilities[:, 1])
    bag.evaluate()
    max_f1 = round(bag.f1_scores[bag.index_of_maximum_f1_score], 2)
    logger.info(f"F1-Score: {max_f1}")

    return max_f1
