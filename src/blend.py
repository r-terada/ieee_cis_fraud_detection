import gc
import os
import json
import click
import sklearn
import warnings
import subprocess
import numpy as np
import pandas as pd
from typing import List, Tuple
from pprint import pformat
from functools import reduce
from attrdict import AttrDict
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from pandas.core.common import SettingWithCopyWarning

import models
import features
from features import ID, JOIN_KEY_COLUMN, TARGET_COLUMN
from utils import init_logger, logger
from utils import read_config, create_oof, create_sub

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


def create_feature(
        feature_config: List[Tuple[str, dict]],
    ) -> pd.DataFrame:
    feats = [
        getattr(features, feature_name)(**feature_params)
        for feature_name, feature_params in feature_config
    ]
    _, feature_te = ID().create_feature()
    for f in feats:
        _, f_te = f.create_feature()
        feature_te = pd.merge(
            feature_te, f_te, how='left', on=JOIN_KEY_COLUMN
        )
    return feature_te


@click.command()
@click.option('--conf_name', type=str, default='lgbm_0000')
def main(conf_name) -> None:

    out_dir = os.path.join('../data/output/', conf_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    open(os.path.join(out_dir, 'log.txt'), 'w').close()  # clear log file
    init_logger(os.path.join(out_dir, 'log.txt'))

    # read config
    conf = read_config(conf_name)

    # create feature
    
    feature_te = create_feature(conf.features)

    logger.info(f"head of features\n{feature_te.head()}")

    model_class = getattr(models, conf.model.name)
    model = model_class(conf.model.model_params, conf.model.fit_params)
    # main
    feats = [c for c in feature_te.columns if c not in conf.cols_to_drop]

    logger.info('start prediction')
    predictions = model.predict(feature_te[feats])

    _, id_te = ID().create_feature()

    logger.info('save submission')
    sub_df = create_sub(id_te, predictions)
    sub_path = os.path.join(out_dir, 'submission.csv')
    sub_df.to_csv(sub_path, index=False)

    logger.info(f"head of submission\n{sub_df.head()}")



if __name__ == "__main__":
    main()
