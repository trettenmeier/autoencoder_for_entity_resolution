import logging

import pandas as pd
from gensim.models.fasttext import FastText as FT_gensim

from ..enums.column_names import IncomingColName


logger = logging.getLogger(__name__)


def make_fasttext_embeddings(df: pd.DataFrame, size_embedding_vector, training_epochs, skip_gram, window):
    embedding_generator = GenerateFasttextEmbeddings(
        df=df,
        size_embedding_vector=size_embedding_vector,
        training_epochs=training_epochs,
        skip_gram=skip_gram,
        window=window
    )
    embedding_generator.generate_sentences_from_data()
    embedding_generator.build_vocabulary()
    embedding_generator.train_model()

    return embedding_generator.model


class GenerateFasttextEmbeddings:
    """
    Trains and saves the fasttext model with the train-part of the 600k dataset.
    Get vector for words with model.wv['word'].
    """

    def __init__(self, df, size_embedding_vector, training_epochs, skip_gram, window):
        self.df = df
        self.size_embedding_vector = size_embedding_vector
        self.training_epochs = training_epochs
        self.skip_gram = skip_gram
        self.window = window

        self.sentences = []

        self.model = FT_gensim(
            sg=self.skip_gram,
            vector_size=self.size_embedding_vector,
            window=self.window,
            epochs=self.training_epochs
        )

    def generate_sentences_from_data(self):
        """
        Reminder: Price is _not_ in there for now.
        """
        logger.info("generating sentences")
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
            IncomingColName.page_internal_info_group.value,
        ]

        for _, datapoint in self.df.iterrows():
            client_sentence = []
            for col_value in client_cols:
                if datapoint[col_value] != "[empty]":
                    client_sentence.extend(datapoint[col_value].split())
            self.sentences.append(client_sentence)

            page_sentence = []
            for col_value in page_cols:
                if datapoint[col_value] != "[empty]":
                    page_sentence.extend(datapoint[col_value].split())
            self.sentences.append(page_sentence)
        logger.info("finished generating sentences")

    def build_vocabulary(self):
        logger.info("building vocabulary")
        self.model.build_vocab(corpus_iterable=self.sentences)
        logger.info("finished building vocabulary")

    def train_model(self):
        logger.info("training model")
        self.model.train(
            corpus_iterable=self.sentences,
            epochs=self.model.epochs,
            total_examples=self.model.corpus_count,
            total_words=self.model.corpus_total_words
        )
        logger.info("finished training model")
