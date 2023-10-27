import numpy as np
import torch
from torch.utils.data import Dataset

from ..enums.column_names import IncomingColName


class MPWordpieceDataset(Dataset):
    def __init__(self, X, wordpiece_tokenizer, number_of_tokens_per_datapoint):
        self.data = X
        self.tokenizer = wordpiece_tokenizer

        self.number_of_tokens_per_datapoint = number_of_tokens_per_datapoint

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        """
        This should return a tokenized tuple of (client, page) that has been truncated and padded to a fixed.
        """
        sentence_a, sentence_b = self.get_sentences_from_df(idx)

        tokenized_a = self.tokenizer.encode(sentence_a).ids
        tokenized_b = self.tokenizer.encode(sentence_b).ids

        length = self.number_of_tokens_per_datapoint
        tokenized_a = self.truncate_and_pad_sentence(tokenized_a, length)
        tokenized_b = self.truncate_and_pad_sentence(tokenized_b, length)

        numpy_a = np.array(tokenized_a, dtype=np.int32)
        numpy_b = np.array(tokenized_b, dtype=np.int32)

        tensor_a = torch.from_numpy(numpy_a)
        tensor_b = torch.from_numpy(numpy_b)

        return tensor_a, tensor_b

    def truncate_and_pad_sentence(self, sentence, length):
        pad_encoding = self.tokenizer.encode("[PAD]").ids[0]
        if len(sentence) == length:
            return sentence
        if len(sentence) > length:
            return sentence[:length]
        if len(sentence) < length:
            for _ in range(length - len(sentence)):
                sentence.append(pad_encoding)
        return sentence

    def get_sentences_from_df(self, idx):
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

        return " ".join(client_sentence), " ".join(page_sentence)


class MPWordpieceDatasetFixedPositions(MPWordpieceDataset):
    def __getitem__(self, idx):
        """
        This should return a tokenized tuple of (client, page) that has been truncated and padded to a fixed length.
        Here, column "positions" are kept after tokenizing.
        """
        tokenized_a, tokenized_b = self._extract_data(idx)

        numpy_a = np.array(tokenized_a, dtype=np.int32)
        numpy_b = np.array(tokenized_b, dtype=np.int32)

        tensor_a = torch.from_numpy(numpy_a)
        tensor_b = torch.from_numpy(numpy_b)

        return tensor_a, tensor_b

    def _extract_data(self, idx):
        # get relative lengths for these 7 fields and make sure these are all integers:
        length = self.number_of_tokens_per_datapoint
        frac = [
            int(2 / length),
            int(4 / length),
            int(4 / length),
            int(0 / length),
            int(3 / length),
            int(12 / length)
        ]  # last element not in yet
        frac.append(length - sum(frac))

        client_cols = {
            IncomingColName.client_oem_info_brand.value: frac[0],
            IncomingColName.client_oem_info_part_type.value: frac[1],
            IncomingColName.client_oem_info_number.value: frac[2],
            "client_info_name": frac[3],
            IncomingColName.client_internal_info_group.value: frac[4],
            "client_info_description": frac[5],
            IncomingColName.client_sales_info_price_sell.value: frac[6]
        }
        page_cols = {
            IncomingColName.page_oem_info_brand.value: frac[0],
            IncomingColName.page_oem_info_part_type.value: frac[1],
            IncomingColName.page_oem_info_number.value: frac[2],
            IncomingColName.page_internal_info_name.value: frac[3],
            IncomingColName.page_internal_info_group.value: frac[4],
            IncomingColName.page_internal_info_description.value: frac[5],
            IncomingColName.page_price.value: frac[6]
        }

        tokenized_a = []
        for key, value in client_cols.items():
            if key == "client_info_name":
                continue
            if key == "client_info_description":
                tokenized_a.extend([self.tokenizer.encode("[PAD]").ids[0] for _ in range(value)])
                continue

            tokenized_a.extend(self._get_tokenized_values_with_fixed_length(idx, key, value))

        tokenized_b = []
        for key, value in page_cols.items():
            if key == IncomingColName.page_internal_info_name.value:
                continue
            if key == IncomingColName.page_oem_info_number.value:
                custom_sentence = []
                custom_sentence.extend(self.data.loc[idx, IncomingColName.page_oem_info_number.value].split())
                custom_sentence.extend(self.data.loc[idx, IncomingColName.page_internal_info_name.value].split())

                if len(custom_sentence) > 0:
                    sentence = " ".join(custom_sentence)
                else:
                    sentence = "[empty]"
                tokenized = self.tokenizer.encode(sentence).ids
                tokenized_b.extend(self.truncate_and_pad_sentence(tokenized, value))
                continue

            tokenized_b.extend(self._get_tokenized_values_with_fixed_length(idx, key, value))

        return tokenized_a, tokenized_b

    def _get_tokenized_values_with_fixed_length(self, idx, col_name, length):
        if self.data.loc[idx, col_name] != "[empty]":
            value = self.data.loc[idx, col_name]
            tokenized = self.tokenizer.encode(value).ids
            fixed_length_tokenized = self.truncate_and_pad_sentence(tokenized, length)
        else:
            fixed_length_tokenized = self.truncate_and_pad_sentence([], length)

        return fixed_length_tokenized
