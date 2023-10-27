import os
import logging
import luigi

from ..tasks.base_task import LuigiBaseTask
from ..utils.get_resources import get_config, get_path_to_working_dir
from ..tasks.preprocessing import PreprocessingTask
from ..tasks.wordpiece_tokenizer import WordpieceTokenizerTask
from ..tasks.classifier_catboost import TrainAndEvaluateClassifierCatboostTask

logger = logging.getLogger(__name__)


class WordpieceCNNAutoencoderCatboostClassifierTask(LuigiBaseTask):
    """
    """
    config = get_config()

    def requires(self):
        return {
            "PreprocessingTask": PreprocessingTask(),
            "WordpieceTokenizerTask": WordpieceTokenizerTask(),
            "TrainAndEvaluateClassifierNeuralNetTask": TrainAndEvaluateClassifierCatboostTask(
                embedding="wordpiece",
                model_class="simple"
            )
        }

    def run(self):
        pass

    def output(self) -> luigi.LocalTarget:
        output_path = os.path.join(
            get_path_to_working_dir(),
            "07_evaluation_results",
            "dummy.dummy"
        )
        return luigi.LocalTarget(output_path)
