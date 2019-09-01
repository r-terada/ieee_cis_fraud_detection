import gc
import os
import json
import click
import warnings
import subprocess
import numpy as np
import pandas as pd
from pprint import pformat
from functools import reduce
from attrdict import AttrDict
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from pandas.core.common import SettingWithCopyWarning

import models
import features
from features import create_feature_pipeline, JOIN_KEY_COLUMN, TARGET_COLUMN
from utils import init_logger, logger
from utils import read_config, create_oof, create_sub

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


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

    # feature_pipeline
    feature_pipe = create_feature_pipeline(conf.features)

    # classifier pipeline
    model_class = getattr(models, conf.model.name)
    if 'kfold_params' in conf.model:
        folds = StratifiedKFold(**conf.model.kfold_params)
        model = models.KFoldModel(
            folds,
            model_class,
            conf.model.model_params,
            conf.model.fit_params,
            conf.get("resample", None)
        )
    else:
        model = models.TrainValSplit(
            model_class,
            conf.model.model_params,
            conf.model.fit_params,
            conf.get("resample", None)
        )

    # main
    full_pipe = Pipeline(steps=[
        ('feature_extraction', feature_pipe),
        ('classification', model)
    ])

    logger.info('read train data')
    train_df = features.read_train()
    target = train_df[TARGET_COLUMN]
    del train_df[TARGET_COLUMN]

    logger.info('start training')
    full_pipe.fit(train_df, target)

    logger.info('read test data')
    test_df = features.read_test()
    logger.info('start prediction')
    predictions = full_pipe.predict(test_df)

    # save
    model = full_pipe.named_steps['classification']

    if hasattr(model, 'results'):
        logger.info('save results')
        result_path = os.path.join(out_dir, 'result.json')
        with open(result_path, "w") as fp:
            json.dump(model.results, fp, indent=2)
        val_score = model.results['trials']['Full']['val_score']
        open(os.path.join(out_dir, f'score_{val_score:.6f}'), 'w').close()

    if hasattr(model, 'oof'):
        logger.info('save oof')
        oof_df = create_oof(train_df, model.oof)
        oof_path = os.path.join(out_dir, 'oof.csv')
        oof_df.to_csv(oof_path, index=False)

    logger.info('save submission')
    sub_df = create_sub(test_df, predictions)
    sub_path = os.path.join(out_dir, 'submission.csv')
    sub_df.to_csv(sub_path, index=False)


if __name__ == "__main__":
    main()
