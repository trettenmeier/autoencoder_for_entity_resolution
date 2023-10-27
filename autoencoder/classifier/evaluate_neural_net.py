import logging
import mlflow
import torch
import numpy as np

from scipy.special import softmax

from ..metrics.metricsbag import MetricsBag


logger = logging.getLogger(__name__)


def evaluate_neural_net_classifier(model, test_loader, y_test):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    predictions = []

    with torch.no_grad():
        for data in test_loader:
            inputs, _ = data

            inputs = inputs.to(device)
            outputs = model(inputs)

            predictions.append(outputs)

    output_list = []
    for tens in predictions:
        temp = (tens.cpu().detach().numpy())
        output_list.append(temp)

    preds = np.concatenate(output_list)
    soft_preds = softmax(preds, axis=1)

    bag = MetricsBag(y_test, soft_preds[:, 1])
    bag.evaluate()
    max_f1 = round(bag.f1_scores[bag.index_of_maximum_f1_score], 2)
    mlflow.log_metric("F1-Score", max_f1)
    logger.info(f"F1-Score: {max_f1}")

    return max_f1
