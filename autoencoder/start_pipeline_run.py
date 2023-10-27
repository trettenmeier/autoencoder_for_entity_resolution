import luigi
import logging
import mlflow

from autoencoder.pipelines.wp__cnn__nnc import WordpieceCNNAutoencoderNeuralNetClassifierTask
from autoencoder.pipelines.wp__rnn__nnc import WordpieceRNNAutoencoderNeuralNetClassifierTask
from autoencoder.pipelines.ft__sa__cbc import FastTextSimpleAutoencoderCatboostClassifierTask
from autoencoder.pipelines.ft__sa__nnc import FastTextSimpleAutoencoderNeuralNetClassifierTask
from autoencoder.pipelines.ft__cnn__nnc import FastTextCNNAutoencoderNeuralNetClassifierTask
from autoencoder.pipelines.ft__rnn__nnc import FasttextRNNAutoencoderNeuralNetClassifierTask
from autoencoder.pipelines.ft_vae_cnn_nnc import FastTextVCNNAutoencoderNeuralNetClassifierTask
from autoencoder.tasks.fasttext_embedding import FasttextEmbeddingTask
from autoencoder.tasks.wordpiece_tokenizer import WordpieceTokenizerTask
from autoencoder.tasks.train_autoencoder import TrainAutoencoderTask
from autoencoder.tasks.classifier_neural_net import TrainAndEvaluateClassifierNeuralNetTask

luigi.interface.core.log_level = "INFO"

logs = logging.getLogger()
logs.setLevel(logging.INFO)
logging.info("Logging initialized!")


if __name__ == '__main__':
    embedding = "fasttext"
    model_class = "vae_cnn"

    # ft = FasttextEmbeddingTask()
    # ft.invalidate()
    ae = TrainAutoencoderTask(embedding=embedding, model_class=model_class)
    ae.invalidate()
    # wp = WordpieceTokenizerTask()
    # wp.invalidate()
    nnc = TrainAndEvaluateClassifierNeuralNetTask(embedding=embedding, model_class=model_class)
    nnc.invalidate()

    mlflow.end_run()
    with mlflow.start_run():
        luigi.build([FastTextVCNNAutoencoderNeuralNetClassifierTask()], workers=1, local_scheduler=True)
