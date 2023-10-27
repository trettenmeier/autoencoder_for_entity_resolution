import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    f1_score,
    precision_score,
    recall_score,
)

if TYPE_CHECKING:
    from typing import Tuple, Iterable

logger = logging.getLogger(__name__)


class MetricsBag:
    """
    This class provides all relevant methods to calculate the precision, recall and f1-scores and is able to
    visualize these metrics.

    Attributes
    ----------
    y: array_like
        Array with ground truth
    y_hat : Array-like
        Array with the probabilities for positive class (aka match)
    thresholds: array_like
        Array with all thresholds that are being checked
    f1_scores: array_like
        Array with all F1-scores for corresponding threshold-index
    index-of_maximum_f1: int
        The index of the element in the thresholds-list that results in the highest F1-score.

    """

    def __init__(self, y_truth: 'Iterable', y_pred_prob: 'Iterable') -> None:
        """
        Calls the actual calculation-methods during initialization of the object.

        Parameters
        ----------
        y_truth : array_like
            Vector with ground truth
        y_pred_prob : array_like
            Vector with predicted probabilities for positive class
        """
        self.y = y_truth
        self.y_hat = y_pred_prob

        self.thresholds = np.linspace(0, 1, num=30)

        self.f1_scores = self._calculate_f1_score()
        self.index_of_maximum_f1_score = np.argmax(self.f1_scores)

    def _calculate_f1_score(self) -> 'np.ndarray':
        """
        Calculates F1-scores for different thresholds in the thresholds-array.

        Returns
        -------
        array
            Array with the F1-scores.
        """

        def calc_func(e: float):
            predictions = [1 if x > e else 0 for x in self.y_hat]
            return f1_score(self.y, predictions)

        results = np.array([calc_func(thresh) for thresh in self.thresholds])

        return results

    def get_max_precision_recall(self) -> 'Tuple[float,float]':
        """
        Fetches the precision and recall values that lead to the highest F1-score.

        Returns
        -------
        tuple (float, float)
            Tuple with (Precision, Recall)
        """
        predictions = [1 if x > self.thresholds[self.index_of_maximum_f1_score] else 0 for x in self.y_hat]
        return precision_score(self.y, predictions), recall_score(self.y, predictions)

    def evaluate(self) -> 'Tuple[plt.Figure,plt.axis]':
        """
        Shows all relevant metrics and numbers in a single plot.

        Returns
        -------
        tuple
            Tuple with matplotlib (Figure, Axes)
        """
        fig = plt.figure(figsize=(20, 4))
        gs = fig.add_gridspec(1, 4)

        ax5 = fig.add_subplot(gs[0, 0])
        self._plot_f1_and_pr_curve(ax5)

        ax6 = fig.add_subplot(gs[0, 1])
        self._plot_f1_key_numbers(ax6)

        ax7 = fig.add_subplot(gs[0, 2])
        ax8 = fig.add_subplot(gs[0, 3])
        self._plot_f1_conf_mat(ax7, ax8)

        return fig, gs

    def _plot_f1_and_pr_curve(self, ax: plt.Axes) -> None:
        """
        Plots the F1-score over all thresholds.

        Parameters
        ----------
        ax: matplotlib.figure.Axes
            The axes that the graph will be drawn on
        """
        ax.set_title("Precision-Recall-Curve and F1-Score")
        PrecisionRecallDisplay.from_predictions(self.y, self.y_hat, ax=ax)
        ax.plot(self.thresholds, self.f1_scores)
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.legend(["Precision-Recall-Curve", "F1-Score over all thresholds"])

    def _plot_f1_key_numbers(self, ax: plt.Axes) -> None:
        """
        Shows key indicators (max. F1-score, Precision, Recall, the best threshold and the number of predictions)

        Parameters
        ----------
        ax: matplotlib.figure.Axes
            The axes that the graph will be drawn on
        """
        max_precision, max_recall = self.get_max_precision_recall()
        text = f"Max. F1-score: {round(self.f1_scores[self.index_of_maximum_f1_score], 2)}\n" \
               f"Precision: {round(max_precision, 2)}\nRecall: {round(max_recall, 2)}" \
               f"\n\nBest threshold: {round(self.thresholds[self.index_of_maximum_f1_score], 2)}"
        ax.text(0.1, 0.3, text, size='xx-large')
        ax.set_title('Precision/Recall combination for max. F1-Score')
        ax.set_xticks([])
        ax.set_yticks([])

    def _plot_f1_conf_mat(self, ax1: plt.Axes, ax2: plt.Axes) -> None:
        """
        Draws 2 confusion matrices with the predictions that result in the highest F1-score. One confusion matrix shows
        the raw numbers the other one is normalized over rows (=all positive labeled datapoints add up to 1 and all
        negative labeled datapoints add up to 1).

        Parameters
        ----------
        ax1: matplotlib.figure.Axes
            The axes that the graph with the raw numbers will be drawn on
        ax2: matplotlib.figure.Axes
            The axes that the graph with normalized numbers will be drawn on
        """
        best_thresh_f1 = self.thresholds[self.index_of_maximum_f1_score]
        predictions = [1 if x > best_thresh_f1 else 0 for x in self.y_hat]

        ConfusionMatrixDisplay.from_predictions(self.y, predictions, ax=ax1)
        ax1.set_title('Confusion Matrix (for F1-score-threshold)')

        ConfusionMatrixDisplay.from_predictions(self.y, predictions, normalize='true', ax=ax2)
        ax2.set_title('Normalized CM (for F1-score-threshold)')
