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


def merge_on_id(df_base, df_additional):
    return pd.merge(df_base, df_additional, on=JOIN_KEY_COLUMN, how='left')
    

def read_train() -> pd.DataFrame:
    df_transaction = pd.read_csv("../data/input/train_transaction.csv")
    df_identity = pd.read_csv("../data/input/train_identity.csv")
    return df_transaction, df_identity


def read_test() -> pd.DataFrame:
    df_transaction = pd.read_csv("../data/input/test_transaction.csv")
    df_identity = pd.read_csv("../data/input/test_identity.csv")
    return df_transaction, df_identity


def read_target() -> pd.DataFrame:
    cache_path = os.path.join(
        CACHE_DIR,
        "target.pkl"
    )
    if os.path.exists(cache_path):
        return pd.read_pickle(cache_path)
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


def agg_group(gr_, feature_name, agg):
    if agg == 'sum':
        feat = gr_[feature_name].sum()
    elif agg == 'mean':
        feat = gr_[feature_name].mean()
    elif agg == 'max':
        feat = gr_[feature_name].max()
    elif agg == 'min':
        feat = gr_[feature_name].min()
    elif agg == 'std':
        feat = gr_[feature_name].std()
    elif agg == 'count':
        feat = gr_[feature_name].count()
    elif agg == 'skew':
        feat = gr_[feature_name].skew()
    elif agg == 'kurt':
        feat = gr_[feature_name].apply(pd.DataFrame.kurt)
    elif agg == 'median':
        feat = gr_[feature_name].median()
    else:
        raise ValueError

    return feat


def add_uids(df):
    df['uid'] = df['card1'].astype(str) + '_' + df['card2'].astype(str)
    df['uid2'] = df['uid'].astype(str) + '_' + df['card3'].astype(str) + '_' + df['card5'].astype(str)
    df['uid3'] = df['uid2'].astype(str) + '_' + df['addr1'].astype(str) + '_' + df['addr2'].astype(str)
    df['uid4'] = df['uid3'].astype(str) + '_' + df['P_emaildomain'].astype(str)
    df['uid5'] = df['uid3'].astype(str) + '_' + df['R_emaildomain'].astype(str)
    df['bank_type'] = df['card3'].astype(str)  + '_' +  df['card5'].astype(str)
    return df


def add_dts(df):
    df['DT'] = df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))
    df['DT_M'] = ((df['DT'].dt.year-2017)*12 + df['DT'].dt.month).astype(np.int8)
    df['DT_W'] = ((df['DT'].dt.year-2017)*52 + df['DT'].dt.weekofyear).astype(np.int8)
    df['DT_D'] = ((df['DT'].dt.year-2017)*365 + df['DT'].dt.dayofyear).astype(np.int16)
    df['DT_hour'] = (df['DT'].dt.hour).astype(np.int8)
    df['DT_day_week'] = (df['DT'].dt.dayofweek).astype(np.int8)
    df['DT_day_month'] = (df['DT'].dt.day).astype(np.int8)
    return df


class Raw:
    train_transaction = None
    train_identity = None
    test_transaction = None
    test_identity = None

    @classmethod
    def read_csvs(cls):
        if cls.train_transaction is None or cls.train_identity is None:
            logger.info(f'[{cls.__name__}] read train.')
            cls.train_transaction, cls.train_identity = read_train()
        if cls.test_transaction is None or cls.test_identity is None:
            logger.info(f'[{cls.__name__}] read test.')
            cls.test_transaction, cls.test_identity = read_test()
        return cls.train_transaction.copy(), cls.train_identity.copy(), cls.test_transaction.copy(), cls.test_identity.copy()


class BaseFeature:
    cache = True

    def __init__(self) -> None:
        pass

    def create_feature(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        fpath_tr = self._train_cache_fpath()
        fpath_te = self._test_cache_fpath()
        if self.cache and os.path.exists(fpath_tr) and os.path.exists(fpath_te):
            self._log("read features from pickled file.")
            return self._read_pickle(fpath_tr), self._read_pickle(fpath_te)
        else:
            self._log(f"no pickled file. create feature.")
            feature_tr, feature_te = self._create_feature()
            feature_tr = reduce_mem_usage(feature_tr)
            feature_te = reduce_mem_usage(feature_te)
            if self.cache:
                self._log(f"save train features to {fpath_tr}")
                self._save_as_pickled_object(feature_tr, fpath_tr)
                self._log(f"save test features to {fpath_te}")
                self._save_as_pickled_object(feature_te, fpath_te)
            self._log("head of feature")
            self._log(f'{feature_tr.head()}\n{feature_te.head()}')
            return feature_tr, feature_te

    def _log(self, message, log_level='info') -> None:
        getattr(logger, log_level)(f'[{self._name}] {message}')

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

    def _create_feature(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError


class ID(BaseFeature):

    def _create_feature(self):
        train_transaction, train_identity, test_transaction, test_identity = Raw.read_csvs()
        return train_transaction[[JOIN_KEY_COLUMN]], test_transaction[[JOIN_KEY_COLUMN]]


class Numerical(BaseFeature):

    def _create_feature(self):
        train_transaction, train_identity, test_transaction, test_identity = Raw.read_csvs()
        train_df = merge_on_id(train_transaction, train_identity)
        test_df = merge_on_id(test_transaction, test_identity)
        numerical_columns = [
            c for c in train_df.columns if (
                train_df[c].dtype != 'object' and c not in [JOIN_KEY_COLUMN, TARGET_COLUMN]
            )
        ]
        return train_df[numerical_columns + [JOIN_KEY_COLUMN]], test_df[numerical_columns + [JOIN_KEY_COLUMN]]


class CategoricalLabelEncode(BaseFeature):

    def _create_feature(self):
        train_transaction, train_identity, test_transaction, test_identity = Raw.read_csvs()
        train_df = merge_on_id(train_transaction, train_identity)
        test_df = merge_on_id(test_transaction, test_identity)
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
        if os.path.exists(os.path.join(out_dir, 'oof.csv')):
            oof_path = os.path.join(out_dir, 'oof.csv')
        else:
            oof_path = os.path.join(out_dir, 'val_prediction.csv')
        sub_path = os.path.join(out_dir, 'submission.csv')
        try:
            self.oof = pd.read_csv(oof_path).rename(columns={TARGET_COLUMN: conf_name})
        except:
            self.oof = None
        self.sub = pd.read_csv(sub_path).rename(columns={TARGET_COLUMN: conf_name})

    def create_feature(self):
        return self.oof, self.sub


class DT_M(BaseFeature):

    def _create_feature(self):
        train_transaction, train_identity, test_transaction, test_identity = Raw.read_csvs()
        for df in [train_transaction, test_transaction]:
            # Temporary
            df['DT'] = df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))
            df['DT_M'] = ((df['DT'].dt.year-2017)*12 + df['DT'].dt.month).astype(np.int8)

        return train_transaction[[JOIN_KEY_COLUMN, 'DT_M']], test_transaction[[JOIN_KEY_COLUMN, 'DT_M']]


class FrequencyEncoding(BaseFeature):
    cache = False

    def __init__(self, column_specs: List[dict]):
        super().__init__()
        self.column_specs = column_specs

    def _create_feature(self):
        feature_tr, feature_te = ID().create_feature()
        for spec in self.column_specs:
            tr, te = ColumnFrequencyEncoder(**spec).create_feature()
            feature_tr = merge_on_id(feature_tr, tr)
            feature_te = merge_on_id(feature_te, te)
        return feature_tr, feature_te


class ColumnFrequencyEncoder(BaseFeature):

    def __init__(self, columns, concat=True, propotion_denominator_columns=None, propotion_only=False):
        super().__init__()
        self.columns = columns
        self.concat = concat
        self.propotion_only = propotion_only
        if self.propotion_only is True:
            assert propotion_denominator_columns is not None
        if propotion_denominator_columns:
            self.propotion_denominator_columns = propotion_denominator_columns
        else:
            self.propotion_denominator_columns = []

    @property
    def _name(self) -> str:
        if self.propotion_denominator_columns:
            if self.propotion_only:
                return f'{self.__class__.__name__}_concat_{self.concat}_propotion_of_{"_".join(self.columns)}_by_{"_".join(self.propotion_denominator_columns)}'
            else:
                return f'{self.__class__.__name__}_concat_{self.concat}_columns_{"_".join(self.columns)}_propotion_of_{"_".join(self.propotion_denominator_columns)}'
        else:
            return f'{self.__class__.__name__}_concat_{self.concat}_columns_{"_".join(self.columns)}'

    def _encode(self, train_df, test_df, columns, new_col_name):
        if self.concat:
            temp_df = pd.concat([train_df, test_df])[columns]
            freq = temp_df.groupby(columns).size().reset_index()
            freq.columns = columns + [new_col_name]
            train_df = pd.merge(train_df, freq, left_on=columns, right_on=columns)
            test_df = pd.merge(test_df, freq, left_on=columns, right_on=columns)
        else:
            for df in [train_df, test_df]:
                freq = df.groupby(columns).size().reset_index()
                freq.columns = columns + [new_col_name]
                df = pd.merge(df, freq, left_on=columns, right_on=columns)

        return train_df, test_df

    def _create_feature(self):
        train_transaction, train_identity, test_transaction, test_identity = Raw.read_csvs()
        train_transaction = add_uids(add_dts(train_transaction))
        test_transaction = add_uids(add_dts(test_transaction))
        cols_to_use = list({JOIN_KEY_COLUMN} | set(self.columns) | set(self.propotion_denominator_columns))
        train_df = merge_on_id(train_transaction, train_identity)[cols_to_use]
        test_df = merge_on_id(test_transaction, test_identity)[cols_to_use]

        freq_col = '_'.join(self.columns) + '_fq_enc'
        train_df, test_df = self._encode(train_df, test_df, self.columns, freq_col)

        ret_cols = [JOIN_KEY_COLUMN]

        if not self.propotion_only:
            ret_cols.append(freq_col)

        if self.propotion_denominator_columns:
            total_col = '_'.join(self.columns) + '_total'
            train_df, test_df = self._encode(train_df, test_df, self.propotion_denominator_columns, total_col)
            propotion_col = '_'.join(self.columns) + '_propotion_by_' + '_'.join(self.propotion_denominator_columns)
            for df in [train_df, test_df]:
                df[propotion_col] = df[freq_col] / df[total_col]
            ret_cols.append(propotion_col)

        return train_df[ret_cols], test_df[ret_cols]


class Aggregation(BaseFeature):
    cache = False
    
    def __init__(self, agg_recipes):
        super().__init__()
        self.agg_recipes = agg_recipes

    def _create_feature(self):
        feature_tr, feature_te = ID().create_feature()

        for groupby_cols, specs in self.agg_recipes:
            for select, agg in specs:
                tr, te = ColumnAggregation(select, agg, groupby_cols).create_feature()
                feature_tr = merge_on_id(feature_tr, tr)
                feature_te = merge_on_id(feature_te, te)
        return feature_tr, feature_te


class ColumnAggregation(BaseFeature):

    def __init__(self, select, agg, groupby_cols):
        super().__init__()
        self.select = select
        self.agg = agg
        self.groupby_cols = groupby_cols

    @property
    def _name(self) -> str:
        return f'{self.select}_{self.agg}_by_{"_".join(self.groupby_cols)}'

    def _create_feature(self):
        train_transaction, train_identity, test_transaction, test_identity = Raw.read_csvs()
        train_transaction = add_uids(add_dts(train_transaction))
        test_transaction = add_uids(add_dts(test_transaction))
        cols_to_use = list({JOIN_KEY_COLUMN} | set(self.groupby_cols) | {self.select})
        train_df = merge_on_id(train_transaction, train_identity)[cols_to_use]
        test_df = merge_on_id(test_transaction, test_identity)[cols_to_use]

        temp_df = pd.concat([train_df, test_df])
        aggregated = temp_df.groupby(self.groupby_cols)[self.select].agg(self.agg).reset_index()
        aggregated.columns = self.groupby_cols + [self._name]

        train_df = pd.merge(train_df, aggregated, left_on=self.groupby_cols, right_on=self.groupby_cols, how='left')
        test_df = pd.merge(test_df, aggregated, left_on=self.groupby_cols, right_on=self.groupby_cols, how='left')

        ret_cols = [JOIN_KEY_COLUMN, self._name]

        return train_df[ret_cols], test_df[ret_cols]




class KonstantinFeature(BaseFeature):
    """
    https://www.kaggle.com/kyakovlev/ieee-gb-2-make-amount-useful-again
    """

    def _create_feature(self):
        train_transaction, train_identity, test_transaction, test_identity = Raw.read_csvs()
        train_df = merge_on_id(train_transaction, train_identity)
        test_df = merge_on_id(test_transaction, test_identity)

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


class KonstantinFeature2(BaseFeature):
    '''
    https://www.kaggle.com/kyakovlev/ieee-fe-with-some-eda/notebook
    '''

    def _create_feature(self):
        train_df, train_identity, test_df, test_identity = Raw.read_csvs()

        logger.debug('checkpoint 1')
        from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
        dates_range = pd.date_range(start='2017-10-01', end='2019-01-01')
        us_holidays = calendar().holidays(start=dates_range.min(), end=dates_range.max())

        # Let's add temporary "time variables" for aggregations
        # and add normal "time variables"
        for df in [train_df, test_df]:
            # Temporary variables for aggregation
            df['DT'] = df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))
            df['DT_M'] = ((df['DT'].dt.year-2017)*12 + df['DT'].dt.month).astype(np.int8)
            df['DT_W'] = ((df['DT'].dt.year-2017)*52 + df['DT'].dt.weekofyear).astype(np.int8)
            df['DT_D'] = ((df['DT'].dt.year-2017)*365 + df['DT'].dt.dayofyear).astype(np.int16)
            df['DT_hour'] = (df['DT'].dt.hour).astype(np.int8)
            df['DT_day_week'] = (df['DT'].dt.dayofweek).astype(np.int8)
            df['DT_day_month'] = (df['DT'].dt.day).astype(np.int8)
            # Possible solo feature
            df['is_december'] = df['DT'].dt.month
            df['is_december'] = (df['is_december']==12).astype(np.int8)
            # Holidays
            df['is_holiday'] = (df['DT'].dt.date.astype('datetime64').isin(us_holidays)).astype(np.int8)

        # Total transactions per timeblock
        for col in ['DT_M','DT_W','DT_D']:
            temp_df = pd.concat([train_df[[col]], test_df[[col]]])
            fq_encode = temp_df[col].value_counts().to_dict()

            train_df[col+'_total'] = train_df[col].map(fq_encode)
            test_df[col+'_total']  = test_df[col].map(fq_encode)

        logger.debug('checkpoint 2')
        ########################### Card columns "outliers"
        for col in ['card1']: 
            valid_card = pd.concat([train_df[[col]], test_df[[col]]])
            valid_card = valid_card[col].value_counts()
            valid_card_std = valid_card.values.std()

            invalid_cards = valid_card[valid_card<=2]
            print('Rare cards',len(invalid_cards))

            valid_card = valid_card[valid_card>2]
            valid_card = list(valid_card.index)

            print('No intersection in Train', len(train_df[~train_df[col].isin(test_df[col])]))
            print('Intersection in Train', len(train_df[train_df[col].isin(test_df[col])]))
            
            train_df[col] = np.where(train_df[col].isin(test_df[col]), train_df[col], np.nan)
            test_df[col]  = np.where(test_df[col].isin(train_df[col]), test_df[col], np.nan)

            train_df[col] = np.where(train_df[col].isin(valid_card), train_df[col], np.nan)
            test_df[col]  = np.where(test_df[col].isin(valid_card), test_df[col], np.nan)
            print('#'*20)

        for col in ['card2','card3','card4','card5','card6',]: 
            print('No intersection in Train', col, len(train_df[~train_df[col].isin(test_df[col])]))
            print('Intersection in Train', col, len(train_df[train_df[col].isin(test_df[col])]))

            train_df[col] = np.where(train_df[col].isin(test_df[col]), train_df[col], np.nan)
            test_df[col]  = np.where(test_df[col].isin(train_df[col]), test_df[col], np.nan)
            print('#'*20)

        logger.debug('checkpoint 3')
        ########################### Client Virtual ID
        # Let's add some kind of client uID based on cardID and addr columns
        # The value will be very specific for each client so we need to remove it
        # from final features. But we can use it for aggregations.
        train_df['uid'] = train_df['card1'].astype(str)+'_'+train_df['card2'].astype(str)
        test_df['uid'] = test_df['card1'].astype(str)+'_'+test_df['card2'].astype(str)

        train_df['uid2'] = train_df['uid'].astype(str)+'_'+train_df['card3'].astype(str)+'_'+train_df['card5'].astype(str)
        test_df['uid2'] = test_df['uid'].astype(str)+'_'+test_df['card3'].astype(str)+'_'+test_df['card5'].astype(str)

        train_df['uid3'] = train_df['uid2'].astype(str)+'_'+train_df['addr1'].astype(str)+'_'+train_df['addr2'].astype(str)
        test_df['uid3'] = test_df['uid2'].astype(str)+'_'+test_df['addr1'].astype(str)+'_'+test_df['addr2'].astype(str)

        train_df['uid4'] = train_df['uid3'].astype(str)+'_'+train_df['P_emaildomain'].astype(str)
        test_df['uid4'] = test_df['uid3'].astype(str)+'_'+test_df['P_emaildomain'].astype(str)

        train_df['uid5'] = train_df['uid3'].astype(str)+'_'+train_df['R_emaildomain'].astype(str)
        test_df['uid5'] = test_df['uid3'].astype(str)+'_'+test_df['R_emaildomain'].astype(str)

        # Add values remove list
        new_columns = ['uid','uid2','uid3','uid4','uid5']

        # Do Global frequency encoding 
        i_cols = ['card1','card2','card3','card5'] + new_columns
        train_df, test_df = self.frequency_encoding(train_df, test_df, i_cols, self_encoding=False)

        logger.debug('checkpoint 4')
        ########################### card3/card5 most common hour 
        # card3 or card5 is a bank country?
        # can we find:
        # - the most popular Transaction Hour
        # - the most popular Week Day
        # and then find distance from it

        # Prepare bank type feature
        for df in [train_df, test_df]:
            df['bank_type'] = df['card3'].astype(str) +'_'+ df['card5'].astype(str)

        encoding_mean = {
            1: ['DT_D','DT_hour','_hour_dist','DT_hour_mean'],
            2: ['DT_W','DT_day_week','_week_day_dist','DT_day_week_mean'],
            3: ['DT_M','DT_day_month','_month_day_dist','DT_day_month_mean'],
        }

        encoding_best = {
            1: ['DT_D','DT_hour','_hour_dist_best','DT_hour_best'],
            2: ['DT_W','DT_day_week','_week_day_dist_best','DT_day_week_best'],
            3: ['DT_M','DT_day_month','_month_day_dist_best','DT_day_month_best'],   
        }

        # Some ugly code here (even worse than in other parts)
        for col in ['card3','card5','bank_type']:
            for df in [train_df, test_df]:
                for encode in encoding_mean:
                    encode = encoding_mean[encode].copy()
                    new_col = col + '_' + encode[0] + encode[2]
                    df[new_col] = df[col].astype(str) +'_'+ df[encode[0]].astype(str)

                    temp_dict = df.groupby([new_col])[encode[1]].agg(['mean']).reset_index().rename(
                                                                            columns={'mean': encode[3]})
                    temp_dict.index = temp_dict[new_col].values
                    temp_dict = temp_dict[encode[3]].to_dict()
                    df[new_col] = df[encode[1]] - df[new_col].map(temp_dict)

                for encode in encoding_best:
                    encode = encoding_best[encode].copy()
                    new_col = col + '_' + encode[0] + encode[2]
                    df[new_col] = df[col].astype(str) +'_'+ df[encode[0]].astype(str)
                    temp_dict = df.groupby([col,encode[0],encode[1]])[encode[1]].agg(['count']).reset_index().rename(
                                                                            columns={'count': encode[3]})

                    temp_dict.sort_values(by=[col,encode[0],encode[3]], inplace=True)
                    temp_dict = temp_dict.drop_duplicates(subset=[col,encode[0]], keep='last')
                    temp_dict[new_col] = temp_dict[col].astype(str) +'_'+ temp_dict[encode[0]].astype(str)
                    temp_dict.index = temp_dict[new_col].values
                    temp_dict = temp_dict[encode[1]].to_dict()
                    df[new_col] = df[encode[1]] - df[new_col].map(temp_dict)

        logger.debug('checkpoint 5')
        ########################### bank_type
        # Tracking nomal activity
        # by doing timeblock frequency encoding
        i_cols = ['bank_type'] #['uid','uid2','uid3','uid4','uid5','bank_type']
        periods = ['DT_M','DT_W','DT_D']

        # We have few options to encode it here:
        # - Just count transactions
        # (but some timblocks have more transactions than others)
        # - Devide to total transactions per timeblock (proportions)
        # - Use both
        # - Use only proportions
        train_df, test_df = self.timeblock_frequency_encoding(train_df, test_df, periods, i_cols, 
                                        with_proportions=False, only_proportions=True)

        logger.debug('checkpoint 6')
        ########################### D Columns
        # From columns description we know that
        # D1-D15: timedelta, such as days between previous transaction, etc.
        # 1. I can't imagine normal negative timedelta values (Let's clip Values)
        # 2. Normalize (Min-Max, Standard score) All D columns, except D1,D2,D9
        # 3. Do some aggregations based on uIDs
        # 4. Freaquency encoding
        # 5. D1,D2 are clipped by max train_df values (let's scale it)
        i_cols = ['D'+str(i) for i in range(1,16)]
        uids = ['uid','uid2','uid3','uid4','uid5','bank_type']
        aggregations = ['mean','std']

        ####### uIDs aggregations
        train_df, test_df = self.uid_aggregation(train_df, test_df, i_cols, uids, aggregations)

        logger.debug('checkpoint 7')
        ####### Cleaning Neagtive values and columns transformations
        for df in [train_df, test_df]:

            for col in i_cols:
                df[col] = df[col].clip(0) 
            
            # Lets transform D8 and D9 column
            # As we almost sure it has connection with hours
            df['D9_not_na'] = np.where(df['D9'].isna(),0,1)
            df['D8_not_same_day'] = np.where(df['D8']>=1,1,0)
            df['D8_D9_decimal_dist'] = df['D8'].fillna(0)-df['D8'].fillna(0).astype(int)
            df['D8_D9_decimal_dist'] = ((df['D8_D9_decimal_dist']-df['D9'])**2)**0.5
            df['D8'] = df['D8'].fillna(-1).astype(int)

        logger.debug('checkpoint 8')
        ####### Values Normalization
        i_cols.remove('D1')
        i_cols.remove('D2')
        i_cols.remove('D9')
        periods = ['DT_D','DT_W','DT_M']
        for df in [train_df, test_df]:
            df = self.values_normalization(df, periods, i_cols)

        for col in ['D1','D2']:
            for df in [train_df, test_df]:
                df[col+'_scaled'] = df[col]/train_df[col].max()

        logger.debug('checkpoint 9')
        ####### Global Self frequency encoding
        # self_encoding=True because 
        # we don't need original values anymore
        i_cols = ['D'+str(i) for i in range(1,16)]
        train_df, test_df = self.frequency_encoding(train_df, test_df, i_cols, self_encoding=True)

        logger.debug('checkpoint 10')
        # Clip Values
        train_df['TransactionAmt'] = train_df['TransactionAmt'].clip(0,5000)
        test_df['TransactionAmt']  = test_df['TransactionAmt'].clip(0,5000)

        # Check if the Transaction Amount is common or not (we can use freq encoding here)
        # In our dialog with a model we are telling to trust or not to these values
        train_df['TransactionAmt_check'] = np.where(train_df['TransactionAmt'].isin(test_df['TransactionAmt']), 1, 0)
        test_df['TransactionAmt_check']  = np.where(test_df['TransactionAmt'].isin(train_df['TransactionAmt']), 1, 0)

        # For our model current TransactionAmt is a noise
        # https://www.kaggle.com/kyakovlev/ieee-check-noise
        # (even if features importances are telling contrariwise)
        # There are many unique values and model doesn't generalize well
        # Lets do some aggregations
        i_cols = ['TransactionAmt']
        uids = ['card1','card2','card3','card5','uid','uid2','uid3','uid4','uid5','bank_type']
        aggregations = ['mean','std']

        # uIDs aggregations
        train_df, test_df = self.uid_aggregation(train_df, test_df, i_cols, uids, aggregations)
 
        logger.debug('checkpoint 11')
        # TransactionAmt Normalization
        periods = ['DT_D','DT_W','DT_M']
        for df in [train_df, test_df]:
            df = self.values_normalization(df, periods, i_cols)

        logger.debug('checkpoint 12')
        # Product type
        train_df['product_type'] = train_df['ProductCD'].astype(str)+'_'+train_df['TransactionAmt'].astype(str)
        test_df['product_type'] = test_df['ProductCD'].astype(str)+'_'+test_df['TransactionAmt'].astype(str)

        i_cols = ['product_type']
        periods = ['DT_D','DT_W','DT_M']
        train_df, test_df = self.timeblock_frequency_encoding(train_df, test_df, periods, i_cols, 
                                                              with_proportions=False, only_proportions=True)
        train_df, test_df = self.frequency_encoding(train_df, test_df, i_cols, self_encoding=True)

        logger.debug('checkpoint 13')
        # Small "hack" to transform distribution 
        # (doesn't affect auc much, but I like it more)
        # please see how distribution transformation can boost your score 
        # (not our case but related)
        # https://scikit-learn.org/stable/auto_examples/compose/plot_transformed_target.html
        train_df['TransactionAmt'] = np.log1p(train_df['TransactionAmt'])
        test_df['TransactionAmt'] = np.log1p(test_df['TransactionAmt'])

        ########################### C Columns
        i_cols = ['C'+str(i) for i in range(1,15)]

        ####### Global Self frequency encoding
        # self_encoding=False because 
        # I want to keep original values
        train_df, test_df = self.frequency_encoding(train_df, test_df, i_cols, self_encoding=False)

        logger.debug('checkpoint 14')
        ####### Clip max values
        for df in [train_df, test_df]:
            for col in i_cols:
                max_value = train_df[train_df['DT_M']==train_df['DT_M'].max()][col].max()
                df[col] = df[col].clip(None,max_value)
        
        for df in [train_identity, test_identity]:
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

        ########################### Merge Identity columns
        temp_df = train_df[['TransactionID']]
        temp_df = temp_df.merge(train_identity, on=['TransactionID'], how='left')
        del temp_df['TransactionID']
        train_df = pd.concat([train_df,temp_df], axis=1)
            
        temp_df = test_df[['TransactionID']]
        temp_df = temp_df.merge(test_identity, on=['TransactionID'], how='left')
        del temp_df['TransactionID']
        test_df = pd.concat([test_df,temp_df], axis=1)

        logger.debug('checkpoint 15')
        i_cols = [
                'DeviceInfo','DeviceInfo_device','DeviceInfo_version',
                'id_30','id_30_device','id_30_version',
                'id_31','id_31_device',
                'id_33',
                ]

        logger.debug('checkpoint 16')
        ####### Global Self frequency encoding
        # self_encoding=True because 
        # we don't need original values anymore
        train_df, test_df = self.frequency_encoding(train_df, test_df, i_cols, self_encoding=True)

        logger.debug('checkpoint 17')
        ########################### ProductCD and M4 Target mean
        # As we already have frequency encoded columns
        # We can have different global transformation on them
        # Target mean?
        # We will transform original values as we don't need them
        # Leakage over folds?
        # Yes, we will have some,
        # But in the same time we already have leakage from 
        # V columns and card1->card6 columns
        # So, no much harm here
        for col in ['ProductCD','M4']:
            temp_dict = train_df.groupby([col])[TARGET_COLUMN].agg(['mean']).reset_index().rename(
                                                                columns={'mean': col+'_target_mean'})
            temp_dict.index = temp_dict[col].values
            temp_dict = temp_dict[col+'_target_mean'].to_dict()

            train_df[col] = train_df[col].map(temp_dict)
            test_df[col]  = test_df[col].map(temp_dict)

        logger.debug('checkpoint 18')
        ########################### Encode Str columns
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

    def values_normalization(self, dt_df, periods, columns):
        for period in periods:
            for col in columns:
                new_col = col +'_'+ period
                dt_df[col] = dt_df[col].astype(float)  

                temp_min = dt_df.groupby([period])[col].agg(['min']).reset_index()
                temp_min.index = temp_min[period].values
                temp_min = temp_min['min'].to_dict()

                temp_max = dt_df.groupby([period])[col].agg(['max']).reset_index()
                temp_max.index = temp_max[period].values
                temp_max = temp_max['max'].to_dict()

                temp_mean = dt_df.groupby([period])[col].agg(['mean']).reset_index()
                temp_mean.index = temp_mean[period].values
                temp_mean = temp_mean['mean'].to_dict()

                temp_std = dt_df.groupby([period])[col].agg(['std']).reset_index()
                temp_std.index = temp_std[period].values
                temp_std = temp_std['std'].to_dict()

                dt_df['temp_min'] = dt_df[period].map(temp_min)
                dt_df['temp_max'] = dt_df[period].map(temp_max)
                dt_df['temp_mean'] = dt_df[period].map(temp_mean)
                dt_df['temp_std'] = dt_df[period].map(temp_std)

                dt_df[new_col+'_min_max'] = (dt_df[col]-dt_df['temp_min'])/(dt_df['temp_max']-dt_df['temp_min'])
                dt_df[new_col+'_std_score'] = (dt_df[col]-dt_df['temp_mean'])/(dt_df['temp_std'])
                del dt_df['temp_min'],dt_df['temp_max'],dt_df['temp_mean'],dt_df['temp_std']
        return dt_df

    def frequency_encoding(self, train_df, test_df, columns, self_encoding=False):
        for col in columns:
            temp_df = pd.concat([train_df[[col]], test_df[[col]]])
            fq_encode = temp_df[col].value_counts(dropna=False).to_dict()
            if self_encoding:
                train_df[col] = train_df[col].map(fq_encode)
                test_df[col]  = test_df[col].map(fq_encode)            
            else:
                train_df[col+'_fq_enc'] = train_df[col].map(fq_encode)
                test_df[col+'_fq_enc']  = test_df[col].map(fq_encode)
        return train_df, test_df

    def timeblock_frequency_encoding(self, train_df, test_df, periods, columns, 
                                     with_proportions=True, only_proportions=False):
        for period in periods:
            for col in columns:
                new_col = col +'_'+ period
                train_df[new_col] = train_df[col].astype(str)+'_'+train_df[period].astype(str)
                test_df[new_col]  = test_df[col].astype(str)+'_'+test_df[period].astype(str)

                temp_df = pd.concat([train_df[[new_col]], test_df[[new_col]]])
                fq_encode = temp_df[new_col].value_counts().to_dict()

                train_df[new_col] = train_df[new_col].map(fq_encode)
                test_df[new_col]  = test_df[new_col].map(fq_encode)

                if only_proportions:
                    train_df[new_col] = train_df[new_col]/train_df[period+'_total']
                    test_df[new_col]  = test_df[new_col]/test_df[period+'_total']

                if with_proportions:
                    train_df[new_col+'_proportions'] = train_df[new_col]/train_df[period+'_total']
                    test_df[new_col+'_proportions']  = test_df[new_col]/test_df[period+'_total']

        return train_df, test_df

    def uid_aggregation(self, train_df, test_df, main_columns, uids, aggregations):
        for main_column in main_columns:  
            for col in uids:
                for agg_type in aggregations:
                    new_col_name = col+'_'+main_column+'_'+agg_type
                    temp_df = pd.concat([train_df[[col, main_column]], test_df[[col,main_column]]])
                    temp_df = temp_df.groupby([col])[main_column].agg([agg_type]).reset_index().rename(
                                                            columns={agg_type: new_col_name})

                    temp_df.index = list(temp_df[col])
                    temp_df = temp_df[new_col_name].to_dict()   

                    train_df[new_col_name] = train_df[col].map(temp_df)
                    test_df[new_col_name]  = test_df[col].map(temp_df)
        return train_df, test_df

    def uid_aggregation_and_normalization(self, train_df, test_df, main_columns, uids, aggregations):
        for main_column in main_columns:  
            for col in uids:

                new_norm_col_name = col+'_'+main_column+'_std_norm'
                norm_cols = []

                for agg_type in aggregations:
                    new_col_name = col+'_'+main_column+'_'+agg_type
                    temp_df = pd.concat([train_df[[col, main_column]], test_df[[col,main_column]]])
                    temp_df = temp_df.groupby([col])[main_column].agg([agg_type]).reset_index().rename(
                                                            columns={agg_type: new_col_name})

                    temp_df.index = list(temp_df[col])
                    temp_df = temp_df[new_col_name].to_dict()   

                    train_df[new_col_name] = train_df[col].map(temp_df)
                    test_df[new_col_name]  = test_df[col].map(temp_df)
                    norm_cols.append(new_col_name)

                train_df[new_norm_col_name] = (train_df[main_column]-train_df[norm_cols[0]])/train_df[norm_cols[1]]
                test_df[new_norm_col_name]  = (test_df[main_column]-test_df[norm_cols[0]])/test_df[norm_cols[1]]          

                del train_df[norm_cols[0]], train_df[norm_cols[1]]
                del test_df[norm_cols[0]], test_df[norm_cols[1]]

        return train_df, test_df

    def check_cor_and_remove(self, train_df, test_df, i_cols, new_columns, remove=False):
        # Check correllation
        print('Correlations','#'*10)
        for col in new_columns:
            cor_cof = np.corrcoef(train_df[TARGET_COLUMN], train_df[col].fillna(0))[0][1]
            print(col, cor_cof)

        if remove:
            print('#'*10)
            print('Best options:')
            best_fe_columns = []
            for main_col in i_cols:
                best_option = ''
                best_cof = 0
                for col in new_columns:
                    if main_col in col:
                        cor_cof = np.corrcoef(train_df[TARGET_COLUMN], train_df[col].fillna(0))[0][1]
                        cor_cof = (cor_cof**2)**0.5
                        if cor_cof>best_cof:
                            best_cof = cor_cof
                            best_option = col

                print(main_col, best_option, best_cof)            
                best_fe_columns.append(best_option)

            for col in new_columns:
                if col not in best_fe_columns:
                    del train_df[col], test_df[col]

        return train_df, test_df


class KonstantinFeature3(BaseFeature):
    '''
    https://www.kaggle.com/kyakovlev/ieee-data-minification
    
    should drop original ['card4', 'card6', 'ProductCD', 'M4', 'id_34', 'id_33']
    '''
    def _create_feature(self):
        train_transaction, train_identity, test_transaction, test_identity = Raw.read_csvs()
        ret_cols = [JOIN_KEY_COLUMN]
        for col in ['card4', 'card6', 'ProductCD', 'M4']:
            ret_cols.append(f'{col}_freq')
            temp_df = pd.concat([train_transaction[[col]], test_transaction[[col]]])
            col_encoded = temp_df[col].value_counts().to_dict()
            train_transaction[f'{col}_freq'] = train_transaction[col].map(col_encoded)
            test_transaction[f'{col}_freq'] = test_transaction[col].map(col_encoded)

        ret_cols = ret_cols + ['id_34_processed', 'id_33_processed', 'id_33_0', 'id_33_1']
        for df in [train_identity, test_identity]:
            df['id_34_processed'] = df['id_34'].fillna(':0')
            df['id_34_processed'] = df['id_34_processed'].apply(lambda x: x.split(':')[1]).astype(np.int8)
            df['id_34_processed'] = np.where(df['id_34_processed']==0, np.nan, df['id_34_processed'])
            df['id_33_processed'] = df['id_33'].fillna('0x0')
            df['id_33_0'] = df['id_33_processed'].apply(lambda x: x.split('x')[0]).astype(int)
            df['id_33_1'] = df['id_33_processed'].apply(lambda x: x.split('x')[1]).astype(int)
            df['id_33_processed'] = np.where(df['id_33_processed']=='0x0', np.nan, df['id_33_processed'])

        train_df = merge_on_id(train_transaction, train_identity)[ret_cols]
        test_df = merge_on_id(test_transaction, test_identity)[ret_cols]

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


class DaysFromBrowserRelease(BaseFeature):

    def _create_feature(self):
        train_transaction, train_identity, test_transaction, test_identity = Raw.read_csvs()
        browser_release_dates = {}
        with open('./misc/browser_release_date.tsv', 'r') as fp:
            for line in fp:
                browser_name, count, release_date = line.split('\t')
                browser_release_dates[browser_name] = pd.to_datetime(release_date)

        for df in [train_identity, test_identity]:
            df['browser_release_date'] = df['id_31'].map(browser_release_dates)

        for df in [train_transaction, test_transaction]:
            df['DT'] = df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))

        train_df = merge_on_id(train_transaction[[JOIN_KEY_COLUMN, 'DT']], train_identity[[JOIN_KEY_COLUMN, 'browser_release_date']])
        test_df = merge_on_id(test_transaction[[JOIN_KEY_COLUMN, 'DT']], test_identity[[JOIN_KEY_COLUMN, 'browser_release_date']])

        for df in [train_df, test_df]:
            df['days_from_browser_release'] = (df.DT - df.browser_release_date).dt.days

        return train_df[[JOIN_KEY_COLUMN, 'days_from_browser_release']], test_df[[JOIN_KEY_COLUMN, 'days_from_browser_release']]


class DaysFromOSRelease(BaseFeature):

    def _create_feature(self):
        train_transaction, train_identity, test_transaction, test_identity = Raw.read_csvs()
        os_release_dates = {}
        with open('./misc/os_release_date.tsv', 'r') as fp:
            for line in fp:
                os_name, count, release_date = line.split('\t')
                os_release_dates[os_name] = pd.to_datetime(release_date)

        for df in [train_identity, test_identity]:
            df['os_release_date'] = df['id_30'].map(os_release_dates)

        for df in [train_transaction, test_transaction]:
            df['DT'] = df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))

        train_df = merge_on_id(train_transaction[[JOIN_KEY_COLUMN, 'DT']], train_identity[[JOIN_KEY_COLUMN, 'os_release_date']])
        test_df = merge_on_id(test_transaction[[JOIN_KEY_COLUMN, 'DT']], test_identity[[JOIN_KEY_COLUMN, 'os_release_date']])

        for df in [train_df, test_df]:
            df['days_from_os_release'] = (df.DT - df.os_release_date).dt.days

        return train_df[[JOIN_KEY_COLUMN, 'days_from_os_release']], test_df[[JOIN_KEY_COLUMN, 'days_from_os_release']]


class OSBrowserReleaseDayDiff(BaseFeature):

    def _create_feature(self):
        train_transaction, train_identity, test_transaction, test_identity = Raw.read_csvs()
        browser_release_dates = {}
        with open('./misc/browser_release_date.tsv', 'r') as fp:
            for line in fp:
                browser_name, count, release_date = line.split('\t')
                browser_release_dates[browser_name] = pd.to_datetime(release_date)

        os_release_dates = {}
        with open('./misc/os_release_date.tsv', 'r') as fp:
            for line in fp:
                os_name, count, release_date = line.split('\t')
                os_release_dates[os_name] = pd.to_datetime(release_date)

        for df in [train_identity, test_identity]:
            df['browser_release_date'] = df['id_31'].map(browser_release_dates)
            df['os_release_date'] = df['id_30'].map(os_release_dates)
            df['os_browser_release_day_diff'] = (df.os_release_date - df.browser_release_date).dt.days

        return train_identity[[JOIN_KEY_COLUMN, 'os_browser_release_day_diff']], test_identity[[JOIN_KEY_COLUMN, 'os_browser_release_day_diff']]


class IDSplit(BaseFeature):

    def _create_feature(self):
        train_transaction, train_identity, test_transaction, test_identity = Raw.read_csvs()
        train_df, test_df = self.id_split(train_identity), self.id_split(test_identity)
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

    def id_split(self, df):
        feat = pd.DataFrame({JOIN_KEY_COLUMN: df[JOIN_KEY_COLUMN]})
        feat['device_name'] = df['DeviceInfo'].str.split('/', expand=True)[0]
        feat['device_version'] = df['DeviceInfo'].str.split('/', expand=True)[1]
        feat['OS_id_30'] = df['id_30'].str.split(' ', expand=True)[0]
        feat['version_id_30'] = df['id_30'].str.split(' ', expand=True)[1]
        feat['browser_id_31'] = df['id_31'].str.split(' ', expand=True)[0]
        feat['version_id_31'] = df['id_31'].str.split(' ', expand=True)[1]
        feat['screen_width'] = df['id_33'].str.split('x', expand=True)[0]
        feat['screen_height'] = df['id_33'].str.split('x', expand=True)[1]
        feat['id_34'] = df['id_34'].str.split(':', expand=True)[1]
        feat['id_23'] = df['id_23'].str.split(':', expand=True)[1]

        feat.loc[feat['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'
        feat.loc[feat['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'
        feat.loc[feat['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'
        feat.loc[feat['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'
        feat.loc[feat['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'
        feat.loc[feat['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'
        feat.loc[feat['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'
        feat.loc[feat['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'
        feat.loc[feat['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'
        feat.loc[feat['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'
        feat.loc[feat['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'
        feat.loc[feat['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'
        feat.loc[feat['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'
        feat.loc[feat['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'
        feat.loc[feat['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'
        feat.loc[feat['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'
        feat.loc[feat['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'

        feat.loc[
            feat.device_name.isin(
                feat.device_name.value_counts()[feat.device_name.value_counts() < 200].index
            ),
            'device_name'
        ] = "Others"

        return feat


class NormalizedEmailDomain(BaseFeature):

    def _create_feature(self):
        train_transaction, train_identity, test_transaction, test_identity = Raw.read_csvs()
        train_df, test_df = self.normalize_domain(train_transaction, test_transaction)
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

    def normalize_domain(self, train_transaction, test_transaction):
        domains = {}
        domains['yahoo'] = ["yahoo.fr", "yahoo.de", "yahoo.es", "yahoo.co.uk", "yahoo.com", "yahoo.com.mx", "ymail.com", "rocketmail.com", "frontiernet.net"]
        domains['ms'] = ["hotmail.com", "live.com.mx", "live.com", "msn.com", "hotmail.es", "outlook.es", "hotmail.fr", "hotmail.de", "hotmail.co.uk"]
        domains['apple'] = ["icloud.com", "mac.com", "me.com"]
        domains['att'] = ["prodigy.net.mx", "att.net", "sbxglobal.net"]
        domains['centulylink'] = ["centurylink.net", "embarqmail.com", "q.com"]
        domains['aol'] = ["aim.com", "aol.com"]
        domains['spectrum'] = ["twc.com", "charter.com"]
        domains['proton'] = ["protonmail.com"]
        domains['comcast'] = ["comcast.net"]
        domains['google'] = ["gmail.com"]
        domains['anonymous'] = ["anonymous.com"]
        domains['N/A'] = [np.nan]

        domain_company_map = {v: k for k, vs in domains.items() for v in vs}

        for df in [train_transaction, test_transaction]:
            df['P_emaildomain_normalized'] = df['P_emaildomain'].astype(str).map(domain_company_map).fillna('other')
            df['R_emaildomain_normalized'] = df['R_emaildomain'].astype(str).map(domain_company_map).fillna('other')

        retcols = [JOIN_KEY_COLUMN, 'P_emaildomain_normalized', 'R_emaildomain_normalized']

        return train_transaction[retcols], test_transaction[retcols]


class RowVColumnsAggregation(BaseFeature):

    def _create_feature(self):
        train_transaction, train_identity, test_transaction, test_identity = Raw.read_csvs()
        v_cols = [c for c in train_transaction if c[0] == 'V']
        for df in [train_transaction, test_transaction]:
            df['v_mean'] = train_transaction[v_cols].mean(axis=1)
            df['v_std'] = train_transaction[v_cols].std(axis=1)
            df['v_max'] = train_transaction[v_cols].max(axis=1)
            df['v_min'] = train_transaction[v_cols].min(axis=1)
        ret_cols = [JOIN_KEY_COLUMN, 'v_mean', 'v_std', 'v_max', 'v_min']
        return train_transaction[ret_cols], test_transaction[ret_cols]


class TransactionAmtAggregation1(BaseFeature):

    def _create_feature(self):
        train_transaction, train_identity, test_transaction, test_identity = Raw.read_csvs()
        ########################### Client Virtual ID
        # Let's add some kind of client uID based on cardID and addr columns
        # The value will be very specific for each client so we need to remove it
        # from final features. But we can use it for aggregations.
        for df in [train_transaction, test_transaction]:
            df['uid'] = df['card1'].astype(str) + '_' + df['card2'].astype(str)
            df['uid2'] = df['uid'].astype(str) + '_' + df['card3'].astype(str) + '_' + df['card5'].astype(str)
            df['uid3'] = df['uid2'].astype(str) + '_' + df['addr1'].astype(str) + '_' + df['addr2'].astype(str)
            df['uid4'] = df['uid3'].astype(str) + '_' + df['P_emaildomain'].astype(str)
            df['uid5'] = df['uid3'].astype(str) + '_' + df['R_emaildomain'].astype(str)
            df['bank_type'] = df['card3'].astype(str)  + '_' +  df['card5'].astype(str)

        ret_cols = [JOIN_KEY_COLUMN]  # feature names
        agg_column = 'TransactionAmt'
        uids = ['card1', 'card2', 'card3', 'card5', 'uid', 'uid2', 'uid3', 'uid4', 'uid5', 'bank_type']
        aggregations = ['max', 'min', 'kurt', 'skew', 'sum', 'count', 'median']  # mean, std isin KonstatinFeature2
        for col in uids:
            gr_ = pd.concat(
                [train_transaction[[col, agg_column]], test_transaction[[col, agg_column]]]
            ).groupby([col])
            for agg_type in aggregations:
                logger.debug(f'calc {agg_type}: {agg_column} group by {col}')
                new_col_name = f'{col}_{agg_column}_{agg_type}'
                ret_cols.append(new_col_name)
                agg_res = agg_group(gr_, agg_column, agg_type).to_dict()

                train_transaction[new_col_name] = train_transaction[col].map(agg_res)
                test_transaction[new_col_name]  = test_transaction[col].map(agg_res)

        return train_transaction[ret_cols], test_transaction[ret_cols]


class TimeToFutureTransaction(BaseFeature):

    def __init__(self, step=-1):
        super().__init__()
        self.step = step

    @property
    def _name(self) -> str:
        return f'{self.__class__.__name__}_step_{self.step}'

    def _create_feature(self):
        train_transaction, train_identity, test_transaction, test_identity = Raw.read_csvs()
        for df in [train_transaction, test_transaction]:
            df['DT'] = df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))

        for df in [train_transaction, test_transaction]:
            df['uid'] = df['card1'].astype(str) + '_' + df['card2'].astype(str)
            df['uid2'] = df['uid'].astype(str) + '_' + df['card3'].astype(str) + '_' + df['card5'].astype(str)
            df['uid3'] = df['uid2'].astype(str) + '_' + df['addr1'].astype(str) + '_' + df['addr2'].astype(str)
            df['uid4'] = df['uid3'].astype(str) + '_' + df['P_emaildomain'].astype(str)
            df['uid5'] = df['uid3'].astype(str) + '_' + df['R_emaildomain'].astype(str)
            df['bank_type'] = df['card3'].astype(str)  + '_' +  df['card5'].astype(str)

        uids = ['card1', 'card2', 'card3', 'card5', 'uid', 'uid2', 'uid3', 'uid4', 'uid5', 'bank_type']

        for df in [train_transaction, test_transaction]:
            for col in uids:
                gr_ = df.groupby([col])
                df[f'{col}_time_to_next_transaction_{self.step}'] = (gr_['DT'].shift(self.step) - gr_['DT'].shift(0)).dt.total_seconds()

        ret_cols = [JOIN_KEY_COLUMN] + [f'{col}_time_to_next_transaction_{self.step}' for col in uids]

        return train_transaction[ret_cols], test_transaction[ret_cols]


class TimeFromPastTransaction(BaseFeature):

    def __init__(self, step=1):
        super().__init__()
        self.step = step

    @property
    def _name(self) -> str:
        return f'{self.__class__.__name__}_step_{self.step}'

    def _create_feature(self):
        train_transaction, train_identity, test_transaction, test_identity = Raw.read_csvs()
        for df in [train_transaction, test_transaction]:
            df['DT'] = df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))

        for df in [train_transaction, test_transaction]:
            df['uid'] = df['card1'].astype(str) + '_' + df['card2'].astype(str)
            df['uid2'] = df['uid'].astype(str) + '_' + df['card3'].astype(str) + '_' + df['card5'].astype(str)
            df['uid3'] = df['uid2'].astype(str) + '_' + df['addr1'].astype(str) + '_' + df['addr2'].astype(str)
            df['uid4'] = df['uid3'].astype(str) + '_' + df['P_emaildomain'].astype(str)
            df['uid5'] = df['uid3'].astype(str) + '_' + df['R_emaildomain'].astype(str)
            df['bank_type'] = df['card3'].astype(str)  + '_' +  df['card5'].astype(str)

        uids = ['card1', 'card2', 'card3', 'card5', 'uid', 'uid2', 'uid3', 'uid4', 'uid5', 'bank_type']

        for df in [train_transaction, test_transaction]:
            for col in uids:
                gr_ = df.groupby([col])
                df[f'{col}_time_from_past_transaction_{self.step}'] = (gr_['DT'].shift(0) - gr_['DT'].shift(self.step)).dt.total_seconds()

        ret_cols = [JOIN_KEY_COLUMN] + [f'{col}_time_from_past_transaction_{self.step}' for col in uids]

        return train_transaction[ret_cols], test_transaction[ret_cols]


class Cents(BaseFeature):

    def __init__(self, round_num=2):
        super().__init__()
        self.round_num = round_num

    @property
    def _name(self) -> str:
        return f'{self.__class__.__name__}_round_by_{self.round_num}'

    def _create_feature(self):
        train_transaction, train_identity, test_transaction, test_identity = Raw.read_csvs()
        for df in [train_transaction, test_transaction]:
            df[f'cents_{self.round_num}'] = np.round(
                df['TransactionAmt'] - np.floor(df['TransactionAmt']),
                self.round_num
            )

        ret_cols = [JOIN_KEY_COLUMN, f'cents_{self.round_num}']

        return train_transaction[ret_cols], test_transaction[ret_cols]


class CentsAsCategory(BaseFeature):

    def __init__(self, round_num=2):
        super().__init__()
        self.round_num = round_num

    @property
    def _name(self) -> str:
        return f'{self.__class__.__name__}_round_by_{self.round_num}'

    def _create_feature(self):
        train_transaction, train_identity, test_transaction, test_identity = Raw.read_csvs()
        for df in [train_transaction, test_transaction]:
            df[f'cents_{self.round_num}_cat'] = np.round(
                df['TransactionAmt'] - np.floor(df['TransactionAmt']),
                self.round_num
            ).mul(10 ** self.round_num).apply(int).astype('category')

        ret_cols = [JOIN_KEY_COLUMN, f'cents_{self.round_num}_cat']

        return train_transaction[ret_cols], test_transaction[ret_cols]


class TransactionAmtDiffFromMean(BaseFeature):

    def __init__(self, concat=False):
        super().__init__()
        self.concat = concat

    @property
    def _name(self) -> str:
        return f'{self.__class__.__name__}_concat_{self.concat}'

    def _create_feature(self):
        train_transaction, train_identity, test_transaction, test_identity = Raw.read_csvs()
        for df in [train_transaction, test_transaction]:
            df['uid'] = df['card1'].astype(str) + '_' + df['card2'].astype(str)
            df['uid2'] = df['uid'].astype(str) + '_' + df['card3'].astype(str) + '_' + df['card5'].astype(str)
            df['uid3'] = df['uid2'].astype(str) + '_' + df['addr1'].astype(str) + '_' + df['addr2'].astype(str)
            df['uid4'] = df['uid3'].astype(str) + '_' + df['P_emaildomain'].astype(str)
            df['uid5'] = df['uid3'].astype(str) + '_' + df['R_emaildomain'].astype(str)
            df['bank_type'] = df['card3'].astype(str)  + '_' +  df['card5'].astype(str)

        uids = ['card1', 'card2', 'card3', 'card5', 'uid', 'uid2', 'uid3', 'uid4', 'uid5', 'bank_type']

        t_amt_means = {}
        if not self.concat:
            for df in [train_transaction, test_transaction]:
                for col in uids:
                    t_amt_means[col] = df.groupby([col]).agg({'TransactionAmt': 'mean'}).to_dict()['TransactionAmt']
        else:
            temp_df = pd.concat([train_transaction, test_transaction])
            for col in uids:
                t_amt_means[col] = temp_df.groupby([col]).agg({'TransactionAmt': 'mean'}).to_dict()['TransactionAmt']

        for df in [train_transaction, test_transaction]:
            for col in uids:
                df[f'TransactionAmt_mean_{col}'] = df[col].map(t_amt_means[col])
                df[f'TransactionAmt_diff_from_mean_{col}'] = df['TransactionAmt'] - df[f'TransactionAmt_mean_{col}']

        ret_cols = [JOIN_KEY_COLUMN] + [f'TransactionAmt_diff_from_mean_{col}' for col in uids]

        return train_transaction[ret_cols], test_transaction[ret_cols]


class TimeToLastTransaction(BaseFeature):

    def __init__(self, concat=True):
        super().__init__()
        self.concat = concat

    @property
    def _name(self) -> str:
        return f'{self.__class__.__name__}_concat_{self.concat}'

    def _create_feature(self):
        train_transaction, train_identity, test_transaction, test_identity = Raw.read_csvs()
        for df in [train_transaction, test_transaction]:
            df['DT'] = df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))

        for df in [train_transaction, test_transaction]:
            df['uid'] = df['card1'].astype(str) + '_' + df['card2'].astype(str)
            df['uid2'] = df['uid'].astype(str) + '_' + df['card3'].astype(str) + '_' + df['card5'].astype(str)
            df['uid3'] = df['uid2'].astype(str) + '_' + df['addr1'].astype(str) + '_' + df['addr2'].astype(str)
            df['uid4'] = df['uid3'].astype(str) + '_' + df['P_emaildomain'].astype(str)
            df['uid5'] = df['uid3'].astype(str) + '_' + df['R_emaildomain'].astype(str)
            df['bank_type'] = df['card3'].astype(str)  + '_' +  df['card5'].astype(str)

        uids = ['card1', 'card2', 'card3', 'card5', 'uid', 'uid2', 'uid3', 'uid4', 'uid5', 'bank_type']

        if self.concat:
            temp_df = pd.concat([train_transaction, test_transaction])
            for col in uids:
                last_transaction = temp_df.groupby([col])['DT'].last().to_dict()
                for df in [train_transaction, test_transaction]:
                    df[f'{col}_last_transaction_time'] = df[col].map(last_transaction)
                    df[f'{col}_time_to_last_transaction'] = (df[f'{col}_last_transaction_time'] - df['DT']).dt.total_seconds()

        else:
            for df in [train_transaction, test_transaction]:
                for col in uids:
                    last_transaction = df.groupby([col])['DT'].last().to_dict()
                    df[f'{col}_last_transaction_time'] = df[col].map(last_transaction)
                    df[f'{col}_time_to_last_transaction'] = (df[f'{col}_last_transaction_time'] - df['DT']).dt.total_seconds()


        ret_cols = [JOIN_KEY_COLUMN] + [f'{col}_time_to_last_transaction' for col in uids]

        return train_transaction[ret_cols], test_transaction[ret_cols]


class NumFollowingTransaction(BaseFeature):

    def __init__(self, concat=True):
        super().__init__()
        self.concat = concat

    @property
    def _name(self) -> str:
        return f'{self.__class__.__name__}_concat_{self.concat}'

    def _create_feature(self):
        train_transaction, train_identity, test_transaction, test_identity = Raw.read_csvs()
        for df in [train_transaction, test_transaction]:
            df['DT'] = df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))

        for df in [train_transaction, test_transaction]:
            df = add_uids(df)

        uids = ['card1', 'card2', 'card3', 'card5', 'uid', 'uid2', 'uid3', 'uid4', 'uid5', 'bank_type']

        if self.concat:
            temp_df = pd.concat([train_transaction, test_transaction])
            temp_df['temp'] = 1
            for col in uids:
                num_following_transaction = temp_df.iloc[::-1].groupby(col)['temp'].cumsum().iloc[::-1] - 1
                train_transaction[f'{col}_num_following_transaction'] = num_following_transaction.iloc[:len(train_transaction)]
                test_transaction[f'{col}_num_following_transaction'] = num_following_transaction.iloc[len(train_transaction):]

        else:
            for df in [train_transaction, test_transaction]:
                df['temp'] = 1
                for col in uids:
                    df[f'{col}_num_following_transaction'] = df.iloc[::-1].groupby(col)['temp'].cumsum().iloc[::-1] - 1


        ret_cols = [JOIN_KEY_COLUMN] + [f'{col}_num_following_transaction' for col in uids]

        return train_transaction[ret_cols], test_transaction[ret_cols]


class DiffVFeatures(BaseFeature):

    def __init__(self, step=1, groupby_col='card1'):
        super().__init__()
        self.step = step
        self.groupby_col = groupby_col

    @property
    def _name(self) -> str:
        return f'{self.__class__.__name__}_step_{self.step}_{self.groupby_col}'

    def _create_feature(self):
        train_transaction, train_identity, test_transaction, test_identity = Raw.read_csvs()
        for df in [train_transaction, test_transaction]:
            df = add_uids(df)

        v_cols = [c for c in train_transaction.columns if c.startswith('V')]

        ret_cols = [JOIN_KEY_COLUMN]
        for df in [train_transaction, test_transaction]:
            gr_ = df.groupby([self.groupby_col])
            for v_col in tqdm(v_cols):
                df[f'{v_col}_diff_{self.step}_by_{self.groupby_col}'] = gr_[v_col].shift(0) - gr_[v_col].shift(self.step)
                ret_cols.append(f'{v_col}_diff_{self.step}_by_{self.groupby_col}')

        return train_transaction[ret_cols], test_transaction[ret_cols]
