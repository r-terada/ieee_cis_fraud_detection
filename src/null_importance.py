import gc
import os
import time
import json
import click
import sklearn
import warnings
import subprocess
import numpy as np
import pandas as pd
import lightgbm as lgb
from pprint import pformat
from functools import reduce
from attrdict import AttrDict
from contextlib import contextmanager
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from pandas.core.common import SettingWithCopyWarning

import models
import features
from features import create_feature, ID, DT_M, read_target, JOIN_KEY_COLUMN, TARGET_COLUMN
from utils import init_logger, logger
from utils import read_config, create_oof, create_sub

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


@contextmanager
def timer(title):
    print(f"== {title}")
    t0 = time.time()
    yield
    print("== done. {:.0f} [s]".format(time.time() - t0))


def get_feature_importances(data, labels, train_features, shuffle, seed=None):
    # Shuffle target if required
    y = labels.copy()
    if shuffle:
        # Here you could as well use a binomial distribution
        y = labels.copy().sample(frac=1.0)

    # Fit LightGBM in RF mode, yes it's quicker than sklearn RandomForest
    dtrain = lgb.Dataset(data[train_features], y, free_raw_data=False, silent=True)
    lgb_params = {
        'objective': 'binary',
        'boosting_type': 'rf',
        'subsample': 0.623,
        'colsample_bytree': 0.7,
        'num_leaves': 255,
        'max_depth': 8,
        'seed': seed,
        'bagging_freq': 1,
        'num_threads': 4,
        'verbose': -1
    }

    # Fit the model
    clf = lgb.train(params=lgb_params, train_set=dtrain, num_boost_round=600)

    # Get feature importances
    imp_df = pd.DataFrame()
    imp_df["feature"] = list(train_features)
    imp_df["importance_gain"] = clf.feature_importance(importance_type='gain')
    imp_df["importance_split"] = clf.feature_importance(importance_type='split')
    imp_df['trn_score'] = roc_auc_score(y, clf.predict(data[train_features]))

    return imp_df


def score_feature_selection(conf, X, y, feats):
    model_class = getattr(models, conf.model.name)
    model = models.LastMonthOut(
        model_class,
        conf.model.model_params,
        conf.model.fit_params,
        conf.get("resample", None),
        conf.model.get('retrain_on_full', True)
    )
    # Fit LightGBM
    model.fit(X, y, feats)
    # Return the last mean / std values
    return model.results['trials']['Full']['val_score']



@click.command()
@click.option('--conf_name', type=str, default='null_imp_000')
def main(conf_name) -> None:

    out_dir = os.path.join('../data/output/', conf_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    open(os.path.join(out_dir, 'log.txt'), 'w').close()  # clear log file
    init_logger(os.path.join(out_dir, 'log.txt'))

    # read config
    conf = read_config(conf_name)

    # create feature
    feature_tr, feature_te = create_feature(conf.features)

    logger.info(f"head of features\n{feature_tr.head()}\n{feature_te.head()}")

    target = read_target()

    # main
    feats = [c for c in feature_tr.columns if c not in conf.cols_to_drop]

    with timer("calc actual importance"):
        if os.path.exists(f"{out_dir}/actual_imp_df.pkl"):
            actual_imp_df = pd.read_pickle(f"{out_dir}/actual_imp_df.pkl")
        else:
            actual_imp_df = get_feature_importances(feature_tr, target, feats, shuffle=False)
            actual_imp_df.to_pickle(f"{out_dir}/actual_imp_df.pkl")

    print(actual_imp_df.head())

    with timer("calc null importance"):
        nb_runs = 100

        if os.path.exists(f"{out_dir}/null_imp_df_run{nb_runs}time.pkl"):
            null_imp_df = pd.read_pickle(f"{out_dir}/null_imp_df_run{nb_runs}time.pkl")
        else:
            null_imp_df = pd.DataFrame()
            for i in range(nb_runs):
                start = time.time()
                # Get current run importances
                imp_df = get_feature_importances(feature_tr, target, feats, shuffle=True)
                imp_df['run'] = i + 1
                # Concat the latest importances with the old ones
                null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)
                # Display current run and time used
                spent = (time.time() - start) / 60
                dsp = '\rDone with %4d of %4d (Spent %5.1f min)' % (i + 1, nb_runs, spent)
                print(dsp, end='', flush=True)
            null_imp_df.to_pickle(f"{out_dir}/null_imp_df_run{nb_runs}time.pkl")

    print(null_imp_df.head())

    with timer('score features'):
        feature_scores = []
        for _f in actual_imp_df['feature'].unique():
            f_null_imps_gain = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
            f_act_imps_gain = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].mean()
            gain_score = np.log(1e-10 + f_act_imps_gain / (1 + np.percentile(f_null_imps_gain, 75)))  # Avoid didvide by zero
            f_null_imps_split = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
            f_act_imps_split = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].mean()
            split_score = np.log(1e-10 + f_act_imps_split / (1 + np.percentile(f_null_imps_split, 75)))  # Avoid didvide by zero
            feature_scores.append((_f, split_score, gain_score))

        scores_df = pd.DataFrame(feature_scores, columns=['feature', 'split_score', 'gain_score'])
        scores_df.to_pickle(f"{out_dir}/feature_scores_df.pkl")

    with timer('calc correlation'):
        correlation_scores = []
        for _f in actual_imp_df['feature'].unique():
            f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
            f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].values
            gain_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
            f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
            f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].values
            split_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
            correlation_scores.append((_f, split_score, gain_score))

        corr_scores_df = pd.DataFrame(correlation_scores, columns=['feature', 'split_score', 'gain_score'])
        corr_scores_df.to_pickle(f"{out_dir}/corr_scores_df.pkl")


    with timer('score feature removal by corr_scores'):
        for threshold in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99][::-1]:
            with open(f"{out_dir}/split_corr_under_threshold_{threshold}.txt", "w") as fp:
                print([_f for _f, _score, _ in correlation_scores if _score < threshold], file=fp)
            with open(f"{out_dir}/gain_corr_under_threshold_{threshold}.txt", "w") as fp:
                print([_f for _f, _, _score in correlation_scores if _score < threshold], file=fp)

        for threshold in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99][::-1]:
            split_feats = [_f for _f, _score, _ in correlation_scores if _score >= threshold]
            gain_feats = [_f for _f, _, _score in correlation_scores if _score >= threshold]

            logger.info('Results for threshold %3d' % threshold)
            logger.info(f'split: use {len(split_feats)} features')
            split_results = score_feature_selection(conf, feature_tr, target, split_feats)
            logger.info(f'\t SPLIT : {split_results}')
            logger.info(f'gain: use {len(gain_feats)} features')
            gain_results = score_feature_selection(conf, feature_tr, target, gain_feats)
            logger.info(f'\t GAIN  : {gain_results}')


if __name__ == "__main__":
    main()
