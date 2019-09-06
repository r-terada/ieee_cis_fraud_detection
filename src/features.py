import os
import datetime
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

START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')


def read_train() -> pd.DataFrame:
    df_identity = pd.read_csv("../data/input/train_identity.csv")
    df_transaction = pd.read_csv("../data/input/train_transaction.csv")
    return pd.merge(df_transaction, df_identity, on=JOIN_KEY_COLUMN, how='left')


def read_test() -> pd.DataFrame:
    df_identity = pd.read_csv("../data/input/test_identity.csv")
    df_transaction = pd.read_csv("../data/input/test_transaction.csv")
    return pd.merge(df_transaction, df_identity, on=JOIN_KEY_COLUMN, how='left')


def read_target() -> pd.DataFrame:
    cache_path = os.path.join(
        CACHE_DIR,
        "target.pkl"
    )
    if os.path.exists(cache_path):
        return pd.read_pickle
    else:
        train_df = read_train()
        target = train_df[TARGET_COLUMN]
        target.to_pickle(cache_path)
        return target


def read_sample_submission() -> pd.DataFrame:
    return pd.read_csv("../data/input/sample_submission.csv")


def create_feature(
        feature_config: List[Tuple[str, dict]],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    features = [
        eval(feature_name)(**feature_params)
        for feature_name, feature_params in feature_config
    ]
    feature_tr, feature_te = ID().create_feature()
    for f in features:
        f_tr, f_te = f.create_feature()
        feature_tr = pd.merge(
            feature_tr, f_tr, how='left', on=JOIN_KEY_COLUMN
        )
        feature_te = pd.merge(
            feature_te, f_te, how='left', on=JOIN_KEY_COLUMN
        )
    return (feature_tr, feature_te)


class Raw:
    train_df = None
    test_df = None

    @classmethod
    def read_csvs(cls):
        if cls.train_df is None:
            logger.info(f'[{cls.__name__}] read train.')
            cls.train_df = read_train()
        if cls.test_df is None:
            logger.info(f'[{cls.__name__}] read test.')
            cls.test_df = read_test()
        return cls.train_df, cls.test_df


class BaseFeature:

    def __init__(self) -> None:
        pass

    def create_feature(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        fpath_tr = self._train_cache_fpath()
        fpath_te = self._test_cache_fpath()
        if os.path.exists(fpath_tr) and os.path.exists(fpath_te):
            self._log("read features from pickled file.")
            return self._read_pickle(fpath_tr), self._read_pickle(fpath_te)
        else:
            self._log(f"no pickled file. create feature.")
            train_df, test_df = Raw.read_csvs()
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


class Prediction:

    def __init__(self, conf_name):
        out_dir = os.path.join('../data/output/', conf_name)
        logger.info(f"[{self.__class__.__name__}] read predictions from {out_dir}")
        oof_path = os.path.join(out_dir, 'oof.csv')
        sub_path = os.path.join(out_dir, 'submission.csv')
        self.oof = pd.read_csv(oof_path).rename(columns={TARGET_COLUMN: conf_name})
        self.sub = pd.read_csv(sub_path).rename(columns={TARGET_COLUMN: conf_name})

    def create_feature(self):
        return self.oof, self.sub


class KonstantinFeature(BaseFeature):
    """
    https://www.kaggle.com/kyakovlev/ieee-gb-2-make-amount-useful-again
    """

    def _create_feature(self, train_df, test_df):
        # D9 and TransactionDT
        for df in [train_df, test_df]:
            # Temporary
            df['DT'] = df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))
            df['DT_M'] = (df['DT'].dt.year-2017)*12 + df['DT'].dt.month
            df['DT_W'] = (df['DT'].dt.year-2017)*52 + df['DT'].dt.weekofyear
            df['DT_D'] = (df['DT'].dt.year-2017)*365 + df['DT'].dt.dayofyear

            df['DT_hour'] = df['DT'].dt.hour
            df['DT_day_week'] = df['DT'].dt.dayofweek
            df['DT_day'] = df['DT'].dt.day

            # D9 column
            df['D9'] = np.where(df['D9'].isna(),0,1)

        # Reset values for "noise" card1
        i_cols = ['card1']
        for col in i_cols:
            valid_card = pd.concat([train_df[[col]], test_df[[col]]])
            valid_card = valid_card[col].value_counts()
            valid_card = valid_card[valid_card>2]
            valid_card = list(valid_card.index)

            train_df[col] = np.where(train_df[col].isin(test_df[col]), train_df[col], np.nan)
            test_df[col]  = np.where(test_df[col].isin(train_df[col]), test_df[col], np.nan)

            train_df[col] = np.where(train_df[col].isin(valid_card), train_df[col], np.nan)
            test_df[col]  = np.where(test_df[col].isin(valid_card), test_df[col], np.nan)

        # M columns (except M4)
        i_cols = ['M1','M2','M3','M5','M6','M7','M8','M9']
        for df in [train_df, test_df]:
            df['M_sum'] = df[i_cols].sum(axis=1).astype(np.int8)
            df['M_na'] = df[i_cols].isna().sum(axis=1).astype(np.int8)

        # ProductCD and M4 Target mean
        for col in ['ProductCD','M4']:
            temp_dict = train_df.groupby([col])[TARGET_COLUMN].agg(['mean']).reset_index().rename(
                                                                columns={'mean': col+'_target_mean'})
            temp_dict.index = temp_dict[col].values
            temp_dict = temp_dict[col+'_target_mean'].to_dict()

            train_df[col+'_target_mean'] = train_df[col].map(temp_dict)
            test_df[col+'_target_mean']  = test_df[col].map(temp_dict)

        # TransactionAmt

        # Let's add some kind of client uID based on cardID ad addr columns
        # The value will be very specific for each client so we need to remove it
        # from final feature. But we can use it for aggregations.
        train_df['uid'] = train_df['card1'].astype(str)+'_'+train_df['card2'].astype(str)
        test_df['uid'] = test_df['card1'].astype(str)+'_'+test_df['card2'].astype(str)

        train_df['uid2'] = train_df['uid'].astype(str)+'_'+train_df['card3'].astype(str)+'_'+train_df['card5'].astype(str)
        test_df['uid2'] = test_df['uid'].astype(str)+'_'+test_df['card3'].astype(str)+'_'+test_df['card5'].astype(str)

        train_df['uid3'] = train_df['uid2'].astype(str)+'_'+train_df['addr1'].astype(str)+'_'+train_df['addr2'].astype(str)
        test_df['uid3'] = test_df['uid2'].astype(str)+'_'+test_df['addr1'].astype(str)+'_'+test_df['addr2'].astype(str)

        # Check if the Transaction Amount is common or not (we can use freq encoding here)
        # In our dialog with a model we are telling to trust or not to these values
        train_df['TransactionAmt_check'] = np.where(train_df['TransactionAmt'].isin(test_df['TransactionAmt']), 1, 0)
        test_df['TransactionAmt_check']  = np.where(test_df['TransactionAmt'].isin(train_df['TransactionAmt']), 1, 0)

        # For our model current TransactionAmt is a noise
        # https://www.kaggle.com/kyakovlev/ieee-check-noise
        # (even if features importances are telling contrariwise)
        # There are many unique values and model doesn't generalize well
        # Lets do some aggregations
        i_cols = ['card1','card2','card3','card5','uid','uid2','uid3']

        for col in i_cols:
            for agg_type in ['mean','std']:
                new_col_name = col+'_TransactionAmt_'+agg_type
                temp_df = pd.concat([train_df[[col, 'TransactionAmt']], test_df[[col,'TransactionAmt']]])
                #temp_df['TransactionAmt'] = temp_df['TransactionAmt'].astype(int)
                temp_df = temp_df.groupby([col])['TransactionAmt'].agg([agg_type]).reset_index().rename(
                                                        columns={agg_type: new_col_name})

                temp_df.index = list(temp_df[col])
                temp_df = temp_df[new_col_name].to_dict()

                train_df[new_col_name] = train_df[col].map(temp_df)
                test_df[new_col_name]  = test_df[col].map(temp_df)

        # Small "hack" to transform distribution
        # (doesn't affect auc much, but I like it more)
        # please see how distribution transformation can boost your score
        # (not our case but related)
        # https://scikit-learn.org/stable/auto_examples/compose/plot_transformed_target.html
        train_df['TransactionAmt'] = np.log1p(train_df['TransactionAmt'])
        test_df['TransactionAmt'] = np.log1p(test_df['TransactionAmt'])

        # 'P_emaildomain' - 'R_emaildomain'
        p = 'P_emaildomain'
        r = 'R_emaildomain'
        uknown = 'email_not_provided'

        for df in [train_df, test_df]:
            df[p] = df[p].fillna(uknown)
            df[r] = df[r].fillna(uknown)

            # Check if P_emaildomain matches R_emaildomain
            df['email_check'] = np.where((df[p]==df[r])&(df[p]!=uknown),1,0)

            df[p+'_prefix'] = df[p].apply(lambda x: x.split('.')[0])
            df[r+'_prefix'] = df[r].apply(lambda x: x.split('.')[0])

        ## Local test doesn't show any boost here,
        ## but I think it's a good option for model stability

        ## Also, we will do frequency encoding later

        # Device info
        for df in [train_df, test_df]:
            ########################### Device info
            df['DeviceInfo'] = df['DeviceInfo'].fillna('unknown_device').str.lower()
            df['DeviceInfo_device'] = df['DeviceInfo'].apply(lambda x: ''.join([i for i in x if i.isalpha()]))
            df['DeviceInfo_version'] = df['DeviceInfo'].apply(lambda x: ''.join([i for i in x if i.isnumeric()]))

            ########################### Device info 2
            df['id_30'] = df['id_30'].fillna('unknown_device').str.lower()
            df['id_30_device'] = df['id_30'].apply(lambda x: ''.join([i for i in x if i.isalpha()]))
            df['id_30_version'] = df['id_30'].apply(lambda x: ''.join([i for i in x if i.isnumeric()]))

            ########################### Browser
            df['id_31'] = df['id_31'].fillna('unknown_device').str.lower()
            df['id_31_device'] = df['id_31'].apply(lambda x: ''.join([i for i in x if i.isalpha()]))

        # Freq encoding
        i_cols = ['card1','card2','card3','card5',
                'C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14',
                'D1','D2','D3','D4','D5','D6','D7','D8',
                'addr1','addr2',
                'dist1','dist2',
                'P_emaildomain', 'R_emaildomain',
                'DeviceInfo','DeviceInfo_device','DeviceInfo_version',
                'id_30','id_30_device','id_30_version',
                'id_31_device',
                'id_33',
                'uid','uid2','uid3',
                ]

        for col in i_cols:
            temp_df = pd.concat([train_df[[col]], test_df[[col]]])
            fq_encode = temp_df[col].value_counts(dropna=False).to_dict()
            train_df[col+'_fq_enc'] = train_df[col].map(fq_encode)
            test_df[col+'_fq_enc']  = test_df[col].map(fq_encode)


        for col in ['DT_M','DT_W','DT_D']:
            temp_df = pd.concat([train_df[[col]], test_df[[col]]])
            fq_encode = temp_df[col].value_counts().to_dict()

            train_df[col+'_total'] = train_df[col].map(fq_encode)
            test_df[col+'_total']  = test_df[col].map(fq_encode)

        periods = ['DT_M','DT_W','DT_D']
        i_cols = ['uid']
        for period in periods:
            for col in i_cols:
                new_column = col + '_' + period

                temp_df = pd.concat([train_df[[col,period]], test_df[[col,period]]])
                temp_df[new_column] = temp_df[col].astype(str) + '_' + (temp_df[period]).astype(str)
                fq_encode = temp_df[new_column].value_counts().to_dict()

                train_df[new_column] = (train_df[col].astype(str) + '_' + train_df[period].astype(str)).map(fq_encode)
                test_df[new_column]  = (test_df[col].astype(str) + '_' + test_df[period].astype(str)).map(fq_encode)

                train_df[new_column] /= train_df[period+'_total']
                test_df[new_column]  /= test_df[period+'_total']

        # Encode Str columns
        # For all such columns (probably not)
        # we already did frequency encoding (numeric feature)
        # so we will use astype('category') here
        for col in list(train_df):
            if train_df[col].dtype=='O':
                print(col)
                train_df[col] = train_df[col].fillna('unseen_before_label')
                test_df[col]  = test_df[col].fillna('unseen_before_label')

                train_df[col] = train_df[col].astype(str)
                test_df[col] = test_df[col].astype(str)

                le = LabelEncoder()
                le.fit(list(train_df[col])+list(test_df[col]))
                train_df[col] = le.transform(train_df[col])
                test_df[col]  = le.transform(test_df[col])

                train_df[col] = train_df[col].astype('category')
                test_df[col] = test_df[col].astype('category')

        return train_df, test_df
