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


def merge_on_id(df_transaction, df_identity):
    return pd.merge(df_transaction, df_identity, on=JOIN_KEY_COLUMN, how='left')
    

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
        return cls.train_transaction, cls.train_identity, cls.test_transaction, cls.test_identity


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
            train_transaction, train_identity, test_transaction, test_identity = Raw.read_csvs()
            feature_tr, feature_te = self._create_feature(train_transaction, train_identity, test_transaction, test_identity)
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

    def _create_feature(self, train_transaction, train_identity, test_transaction, test_identity) -> Tuple[pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError


class ID(BaseFeature):

    def _create_feature(self, train_transaction, train_identity, test_transaction, test_identity):
        return train_transaction[[JOIN_KEY_COLUMN]], test_transaction[[JOIN_KEY_COLUMN]]


class Numerical(BaseFeature):

    def _create_feature(self, train_transaction, train_identity, test_transaction, test_identity):
        train_df = merge_on_id(train_transaction, train_identity)
        test_df = merge_on_id(test_transaction, test_identity)
        numerical_columns = [
            c for c in train_df.columns if (
                train_df[c].dtype != 'object' and c not in [JOIN_KEY_COLUMN, TARGET_COLUMN]
            )
        ]
        return train_df[numerical_columns + [JOIN_KEY_COLUMN]], test_df[numerical_columns + [JOIN_KEY_COLUMN]]


class CategoricalLabelEncode(BaseFeature):

    def _create_feature(self, train_transaction, train_identity, test_transaction, test_identity):
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
        oof_path = os.path.join(out_dir, 'oof.csv')
        sub_path = os.path.join(out_dir, 'submission.csv')
        self.oof = pd.read_csv(oof_path).rename(columns={TARGET_COLUMN: conf_name})
        self.sub = pd.read_csv(sub_path).rename(columns={TARGET_COLUMN: conf_name})

    def create_feature(self):
        return self.oof, self.sub


class DT_M(BaseFeature):

    def _create_feature(self, train_transaction, train_identity, test_transaction, test_identity):
        for df in [train_transaction, test_transaction]:
            # Temporary
            df['DT'] = df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))
            df['DT_M'] = ((df['DT'].dt.year-2017)*12 + df['DT'].dt.month).astype(np.int8)

        return train_transaction[[JOIN_KEY_COLUMN, 'DT_M']], test_transaction[[JOIN_KEY_COLUMN, 'DT_M']]


class KonstantinFeature(BaseFeature):
    """
    https://www.kaggle.com/kyakovlev/ieee-gb-2-make-amount-useful-again
    """

    def _create_feature(self, train_transaction, train_identity, test_transaction, test_identity):
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

    def _create_feature(self, train_transaction, train_identity, test_transaction, test_identity):
        train_df = train_transaction
        test_df = test_transaction

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


class DaysFromBrowserRelease(BaseFeature):

    def _create_feature(self, train_transaction, train_identity, test_transaction, test_identity):
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

    def _create_feature(self, train_transaction, train_identity, test_transaction, test_identity):
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

    def _create_feature(self, train_transaction, train_identity, test_transaction, test_identity):
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
