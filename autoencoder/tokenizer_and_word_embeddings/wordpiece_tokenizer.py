import logging
import numpy as np

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer

from ..enums.column_names import IncomingColName
from ..utils.get_resources import get_config


def wordpiece_tokenizer(X_train):
    config = get_config()

    tokenizer = WordPieceTokenizer(X_train, config["wordpiece_tokenizer"]["vocab_size"])
    tokenizer.train()
    return tokenizer.tokenizer


class WordPieceTokenizer:
    CORPUS_PATH = "temp_wordpiece_corpus.txt"

    def __init__(self, X_train, vocab_size):
        self.X_train = X_train
        self.vocab_size = vocab_size

        self.tokenizer = Tokenizer(WordPiece())
        self.tokenizer.pre_tokenizer = Whitespace()

        self.corpus = None

    def _create_vocabulary(self):
        logging.info("Creating corpus for tokenizer ONLY on train-split.")

        self.corpus = WordPieceTokenizer._extract_sentences_from_df(self.X_train)

        with open(self.CORPUS_PATH, 'w') as f:
            for sentence in self.corpus:
                f.write(sentence)
                f.write("\n\n")

        logging.info("Done creating corpus.")

    @staticmethod
    def _extract_sentences_from_df(df):
        client_cols = [
            IncomingColName.client_oem_info_brand.value,
            IncomingColName.client_oem_info_number.value,
            IncomingColName.client_oem_info_part_type.value,
            IncomingColName.client_internal_info_group.value
        ]

        page_cols = [
            IncomingColName.page_oem_info_brand.value,
            IncomingColName.page_oem_info_number.value,
            IncomingColName.page_oem_info_part_type.value,
            IncomingColName.page_internal_info_name.value,
            IncomingColName.page_internal_info_description.value,
            IncomingColName.page_internal_info_group.value
        ]

        def page_concat(x):
            return " ".join([x[i] for i in page_cols])

        def client_concat(x):
            return " ".join([x[i] for i in client_cols])

        df["page_concat"] = df.apply(page_concat, axis=1)
        df["client_concat"] = df.apply(client_concat, axis=1)

        train_page_sentences = df["page_concat"].values
        train_client_sentences = df["client_concat"].values

        return np.concatenate([train_client_sentences, train_page_sentences])

    def train(self):
        self._create_vocabulary()

        logging.info("Creating vocab.")
        trainer = WordPieceTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["[PAD]", "[UNK]", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        )
        self.tokenizer.train(files=[self.CORPUS_PATH], trainer=trainer)
        logging.info("Done creating vocab.")
