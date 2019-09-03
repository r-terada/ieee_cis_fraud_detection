import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Union, List, Tuple
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold

from utils import logger, reduce_mem_usage


CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '../data/cache/features/'
)
TARGET_COLUMN = "isFraud"
JOIN_KEY_COLUMN = "TransactionID"


def read_train() -> pd.DataFrame:
    df_identity = pd.read_csv("../data/input/train_identity.csv")
    df_transaction = pd.read_csv("../data/input/train_transaction.csv")
    return pd.merge(df_transaction, df_identity, on=JOIN_KEY_COLUMN, how='left')


def read_test() -> pd.DataFrame:
    df_identity = pd.read_csv("../data/input/test_identity.csv")
    df_transaction = pd.read_csv("../data/input/test_transaction.csv")
    return pd.merge(df_transaction, df_identity, on=JOIN_KEY_COLUMN, how='left')


def read_sample_submission() -> pd.DataFrame:
    return pd.read_csv("../data/input/sample_submission.csv")


def create_feature(
        feature_config: List[Tuple[str, dict]],
        train_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    features = [
        eval(feature_name)(**feature_params)
        for feature_name, feature_params in feature_config
    ]
    feature_tr, feature_te = ID().create_feature(train_df, test_df)
    for f in features:
        f_tr, f_te = f.create_feature(train_df, test_df)
        feature_tr = pd.merge(
            feature_tr, f_tr, how='left', on=JOIN_KEY_COLUMN
        )
        feature_te = pd.merge(
            feature_te, f_te, how='left', on=JOIN_KEY_COLUMN
        )
    return (feature_tr, feature_te)


class BaseFeature:

    def __init__(self) -> None:
        pass

    def create_feature(self, train_df, test_df) -> Tuple[pd.DataFrame, pd.DataFrame]:
        fpath_tr = self._train_cache_fpath()
        fpath_te = self._test_cache_fpath()
        if os.path.exists(fpath_tr) and os.path.exists(fpath_te):
            self._log("read features from pickled file.")
            return self._read_pickle(fpath_tr), self._read_pickle(fpath_te)
        else:
            self._log(f"no pickled file. create feature.")
            feature_tr, feature_te = self._create_feature(train_df, test_df)
            feature_tr = reduce_mem_usage(feature_tr)
            feature_te = reduce_mem_usage(feature_te)
            self._log(f"save train features to {fpath_tr}")
            self._save_as_pickled_object(feature_tr, fpath_tr)
            self._log(f"save test features to {fpath_te}")
            self._save_as_pickled_object(feature_te, fpath_te)
            self._log("head of feature", 'debug')
            self._log(feature_tr.head(), 'debug')
            return feature_tr, feature_te

    def _log(self, message, log_level='info') -> None:
        getattr(logger, log_level)(f'[{self.__class__.__name__}] {message}')

    @property
    def _name(self) -> str:
        return self.__class__.__name__

    def _train_cache_fpath(self) -> str:
        return os.path.join(
            CACHE_DIR,
            f"{self._name}_train.pkl"
        )

    def _test_cache_fpath(self) -> str:
        return os.path.join(
            CACHE_DIR,
            f"{self._name}_test.pkl"
        )

    def _read_pickle(self, pkl_fpath) -> pd.DataFrame:
        return pd.read_pickle(pkl_fpath)

    def _save_as_pickled_object(self, df, pkl_fpath) -> None:
        df.to_pickle(pkl_fpath, compression=None)

    def _create_feature(self, train_df, test_df) -> Tuple[pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError


class ID(BaseFeature):

    def _create_feature(self, train_df, test_df):
        return train_df[[JOIN_KEY_COLUMN]], test_df[[JOIN_KEY_COLUMN]]


class Numerical(BaseFeature):

    def _create_feature(self, train_df, test_df):
        numerical_columns = [
            c for c in train_df.columns if (
                train_df[c].dtype != 'object' and c not in [JOIN_KEY_COLUMN, TARGET_COLUMN]
            )
        ]
        return train_df[numerical_columns + [JOIN_KEY_COLUMN]], test_df[numerical_columns + [JOIN_KEY_COLUMN]]


class CategoricalLabelEncode(BaseFeature):

    def _create_feature(self, train_df, test_df):
        categorical_columns = [
            c for c in train_df.columns if (
                train_df[c].dtype == 'object' and c not in [JOIN_KEY_COLUMN, TARGET_COLUMN]
            )
        ]
        for col in tqdm(categorical_columns, desc=self.__class__.__name__):
            train_df[col] = train_df[col].fillna('N/A')
            test_df[col]  = test_df[col].fillna('N/A')

            train_df[col] = train_df[col].astype(str)
            test_df[col] = test_df[col].astype(str)

            le = LabelEncoder()
            le.fit(list(train_df[col]) + list(test_df[col]))
            train_df[col] = le.transform(train_df[col])
            test_df[col]  = le.transform(test_df[col])

            train_df[col] = train_df[col].astype('category')
            test_df[col] = test_df[col].astype('category')
        return train_df[categorical_columns + [JOIN_KEY_COLUMN]], test_df[categorical_columns + [JOIN_KEY_COLUMN]]
