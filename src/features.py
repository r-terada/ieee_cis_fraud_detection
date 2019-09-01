import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Union, Dict
from category_encoders import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import TransformerMixin, BaseEstimator

from utils import logger
from pipeline_helpers import ColumnUnSelector, PandasFeatureUnion, PrintInfo


TARGET_COLUMN = "isFraud"
JOIN_KEY_COLUMN = "TransactionID"
COLUMNS_UNUSE_TO_PREDICT = ["TransactionID", "isFraud", "TransactionDT"]


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


def create_feature_pipeline(feature_config: Dict[str, dict]):
    pipe = make_pipeline(
        ColumnUnSelector(COLUMNS_UNUSE_TO_PREDICT),
        PandasFeatureUnion([
            (
                f'{feature_name} {feature_params}',
                eval(feature_name)(**feature_params)
            )
            for feature_name, feature_params in feature_config
        ], n_jobs=1),
        PrintInfo(),
    )
    return pipe


class BaseFeature(BaseEstimator, TransformerMixin):

    def __init__(self) -> None:
        pass

    def fit(self, X, y=None):
        logger.info(f'{self.__class__.__name__}: fit')
        self._fit_impl(X, y)
        return self

    def _fit_impl(self, X, y):
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info(f'{self.__class__.__name__}: transform')
        return self._transform_impl(X)

    def _transform_impl(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class Numerical(BaseFeature):

    def _transform_impl(self, X: pd.DataFrame) -> pd.DataFrame:
        numerical_columns = [
            c for c in X.columns if X[c].dtype != 'object'
        ]
        return X[numerical_columns]


class CategoricalLabelEncode(BaseFeature):

    def __init__(self) -> None:
        self.encoder = OrdinalEncoder(return_df=True)

    def _fit_impl(self, X, y):
        categorical_columns = [
            c for c in X.columns if X[c].dtype == 'object'
        ]
        self.encoder.fit(X[categorical_columns])

    def _transform_impl(self, X: pd.DataFrame) -> pd.DataFrame:
        categorical_columns = [
            c for c in X.columns if X[c].dtype == 'object'
        ]
        return self.encoder.transform(X[categorical_columns])


class Prediction(BaseFeature):

    def __init__(self, conf_name):
        out_dir = os.path.join('../data/output/', conf_name)
        oof_path = os.path.join(out_dir, 'oof.csv')
        sub_path = os.path.join(out_dir, 'submission.csv')
        self.oof = pd.read_csv(oof_path).drop(JOIN_KEY_COLUMN, axis=1).rename(columns={TARGET_COLUMN: conf_name})
        self.sub = pd.read_csv(sub_path).drop(JOIN_KEY_COLUMN, axis=1).rename(columns={TARGET_COLUMN: conf_name})

    # override
    # fit_transform is called only at training time
    def fit_transform(self, X, y) -> pd.DataFrame:
        return self.oof

    # override
    # test time
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.sub
