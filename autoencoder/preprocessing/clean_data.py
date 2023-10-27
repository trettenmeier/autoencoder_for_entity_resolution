from typing import Callable
import psutil
import logging
import concurrent.futures
import unicodedata
import re
from collections import OrderedDict
import pandas as pd
import numpy as np

from ..enums.column_names import IncomingColName


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    logger.info("cleaning data")
    df = DataCleaner.parallelize(DataCleaner.do_cleaning, df)
    logger.info("done cleaning data")

    return df


class DataCleaner:
    """
    Data cleaning with these steps:
        * (optional) renaming columns according to our 'standard-scheme'
        * setting dtype of columns to str
        * all characters to lowercase
        * remove or replace some characters like "_", ",", etc. according to a set of rules
    """
    cols = [
        IncomingColName.client_oem_info_brand.value,
        IncomingColName.client_oem_info_number.value,
        IncomingColName.client_oem_info_part_type.value,
        IncomingColName.client_sales_info_price_sell.value,
        IncomingColName.client_internal_info_group.value,
        IncomingColName.page_oem_info_brand.value,
        IncomingColName.page_oem_info_number.value,
        IncomingColName.page_oem_info_part_type.value,
        IncomingColName.page_internal_info_name.value,
        IncomingColName.page_price.value,
        IncomingColName.page_internal_info_description.value,
        IncomingColName.page_internal_info_group.value,
    ]
    str_cols = cols

    @staticmethod
    def parallelize(transformer_method: 'Callable', X: pd.DataFrame) -> pd.DataFrame:
        """
        Parallelizes only when there are at least 300 data points (arbitrarily chosen). If there are more processes than
        rows in the df we get an error because there will be empty dfs.

        Parameters
        ----------
        transformer_method: CustomerTransformer
            the transformer to be parallelized
        X: DataFrame
            the dataframe to be transformed

        Returns
        -------
        DataFrame
            full dataframe with transformer applied
        """
        num_datapoints_before = X.shape[0]

        num_procs = psutil.cpu_count() - 1
        if num_procs == 0:
            num_procs = 1

        if num_procs > X.shape[0]:
            num_procs = X.shape[0]

        if X.shape[0] > 300:
            splits_df = np.array_split(X, num_procs)
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_procs) as executor:
                results = executor.map(transformer_method, splits_df)
            result = pd.concat(results)

        else:
            result = transformer_method(X)

        assert num_datapoints_before == result.shape[0], "Error. Missing rows after transformer was applied!"
        return result

    @staticmethod
    def do_cleaning(X: pd.DataFrame) -> pd.DataFrame:
        """
        Formats the entire dataframe so everything needed for the model is present and in the correct form.
        """
        X = DataCleaner.apply_correct_columns(X)
        X = DataCleaner.make_df_value_adjustments(X)
        X = DataCleaner.fill_null_values(X)
        return X

    @staticmethod
    def apply_correct_columns(X: pd.DataFrame) -> pd.DataFrame:
        """
        Adds in missing columns and removes ones that are not necessary.
        """
        cols = DataCleaner.cols

        for column in cols:
            if column not in X:
                X[column] = np.nan

        X = X.loc[:, cols]
        return X

    @staticmethod
    def make_df_value_adjustments(X: pd.DataFrame) -> pd.DataFrame:
        """
        Makes all values in the dataframe of type String and any empty values np.nan.
        """
        str_cols = DataCleaner.str_cols
        X[str_cols] = X[str_cols].astype(str)

        for col in str_cols:
            X[col] = X[col].str.lower()
            X[col] = X[col].apply(lambda x: DataCleaner.clean_string(x))

        X[str_cols] = X[str_cols].replace({
            "": np.nan,
            " ": np.nan,
            "nan": np.nan,
            "none": np.nan
        })
        return X

    @staticmethod
    def fill_null_values(X: pd.DataFrame) -> pd.DataFrame:
        """
        Fills nan values with spaces
        """
        X = X.fillna(value="[empty]")
        return X

    @staticmethod
    def clean_string(x: str) -> str:
        """
        Does string specific cleaning - such as normalizing and removing irregular characters

        See Also
        --------
        get_rules: for specific misc string rules that should be applied for our model
        """
        x = DataCleaner.discard_ignored_pieces(x)
        x = DataCleaner.normalize_caseless(x)
        x = x.encode('ascii', errors='ignore').decode()

        rules = DataCleaner.get_rules()
        for key in rules.keys():
            x = re.sub(key, rules[key], x)

        return x

    @staticmethod
    def normalize_caseless(text: str) -> str:
        """
        Normalizes input string by stripping and/or replacing special characters.
        See https://stackoverflow.com/a/29247821

        Parameters
        ----------
        text: str
            string to normalize.

        Returns
        -------
        str
            normalized string.
        """
        special_char_map = {ord('ä'): 'ae', ord('ü'): 'ue', ord('ö'): 'oe', ord('ß'): 'ss', ord('Ä'): 'Ae',
                            ord('Ü'): 'Ue', ord('Ö'): 'Oe'}
        text = str(text).translate(special_char_map)
        return unicodedata.normalize("NFKD", str(text).casefold())

    @staticmethod
    def discard_ignored_pieces(brand_name: str) -> str:
        """
        Removes all unwanted words from Brand name (e.g. Vertrieb, GmbH, Inc, etc.)
        Check brand_pieces_ignore for a complete list
        """
        return DataCleaner.get_ignore_words_from_brands_regex().sub("", brand_name) if brand_name and str(
            brand_name) != "nan" else None

    @staticmethod
    def get_ignore_words_from_brands_regex():
        return re.compile(
            r"[^a-zA-Z0-9](" + "|".join(DataCleaner.get_words_to_ignore_in_brands())
            + ")(?!=[a-zA-Z0-9])", re.RegexFlag.IGNORECASE)

    @staticmethod
    def get_words_to_ignore_in_brands():
        return [
                "Industry",
                "Inc",
                "GmbH",
                "Co",
                "KG",
                "AG",
                "eK",
                "OHG",
                "Ltd",
                "&",
                "\\+"]

    @staticmethod
    def get_rules() -> 'OrderedDict[str, str]':
        """
        Rules for replacing . - random chars / _ and unnecessary spaces from the strings.
        """
        rules = OrderedDict()

        rules[r"\. "] = " "
        rules[r"\."] = ""

        rules[r"([0-9])\-([0-9])"] = r"\g<1>\g<2>"
        rules[r"([a-zA-Z])\-([a-zA-Z])"] = r"\g<1> \g<2>"
        rules[r"\-"] = ""

        rules[r"\+|\(|\)|\#|\&|\,|×|\|;"] = " "

        rules[r"_"] = " "

        rules[r"([a-zA-Z])\/([a-zA-Z])"] = r"\g<1> \g<2>"
        rules[r"\/"] = ""

        rules[r"  +"] = " "
        rules[r" $"] = ""

        return rules
