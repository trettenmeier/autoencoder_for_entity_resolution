import torch
import numpy as np

from torch.utils.data import Dataset

from ..enums.column_names import IncomingColName


class MPFasttextDataset(Dataset):
    def __init__(self, df, fasttext_model, truncate_vectors_to_x_words, size_embedding_vector):
        self.data = df
        self.fasttext_model = fasttext_model
        self.truncate_vectors_to_x_words = truncate_vectors_to_x_words
        self.size_embedding_vector = size_embedding_vector

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # there is no need to return the label. we only have matches in the filtered dataset.
        # label = self.label.iloc[idx]
        embedding_a, embedding_b = self.get_word_embeddings(idx)

        tensor_embedding_a = torch.from_numpy(embedding_a)
        tensor_embedding_b = torch.from_numpy(embedding_b)

        return tensor_embedding_a, tensor_embedding_b

    def get_word_embeddings(self, idx):
        sentence_a, sentence_b = self.get_tokenized_sentences(idx)

        # truncate to static length, pad if necessary
        length = self.truncate_vectors_to_x_words
        sentence_a = self.truncate_and_pad_sentence(sentence_a, length)
        sentence_b = self.truncate_and_pad_sentence(sentence_b, length)

        return self.get_embedding_from_list(sentence_a), self.get_embedding_from_list(sentence_b)

    def get_embedding_from_list(self, sentence):
        for i in range(len(sentence)):
            if sentence[i] == "[padding]":
                sentence[i] = np.zeros(self.size_embedding_vector)
                continue
            sentence[i] = self.fasttext_model.wv[sentence[i]]
        return np.array(sentence, dtype=np.float32)

    def truncate_and_pad_sentence(self, sentence, length):
        if len(sentence) == length:
            return sentence
        if len(sentence) > length:
            return sentence[:length]
        if len(sentence) < length:
            for _ in range(length - len(sentence)):
                sentence.append("[padding]")
        return sentence

    def get_tokenized_sentences(self, idx):
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
            IncomingColName.page_internal_info_group.value,
            IncomingColName.page_internal_info_description.value,
        ]

        client_sentence = []
        for col_name in client_cols:
            if self.data.loc[idx, col_name] != "[empty]":
                client_sentence.extend(self.data.loc[idx, col_name].split())

        page_sentence = []
        for col_name in page_cols:
            if self.data.loc[idx, col_name] != "[empty]":
                page_sentence.extend(self.data.loc[idx, col_name].split())

        return client_sentence, page_sentence
