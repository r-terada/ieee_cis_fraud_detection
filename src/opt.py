import os
import copy
import time
import json
import click
import pickle
import sklearn
import warnings
import hyperopt
import numpy as np
import pandas as pd
from pprint import pformat
from pandas.core.common import SettingWithCopyWarning
from hyperopt import fmin, tpe, hp, STATUS_OK

import models
import features
from features import create_feature, ID, DT_M, read_target, JOIN_KEY_COLUMN, TARGET_COLUMN
from utils import init_logger, logger
from utils import read_config, create_oof, create_sub

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


NUM_EVAL = 200


def opt(feature_tr, target, out_dir, conf):

    model_class = getattr(models, conf.model.name)
    feats = [c for c in feature_tr.columns if c not in conf.cols_to_drop]

    search_space = {
        k: hp.quniform(k, v["min"], v["max"], v["step"]) for k, v in conf.model.search_space.items()
    }
    param_types = {k: v['type'] for k, v in conf.model.search_space.items()}

    opt_idx = 0

    def objective(opt_params):
        nonlocal opt_idx
        opt_idx += 1
        st_time = time.time()

        logger.info(f'========== trial {opt_idx:0>3}')
        model_class = getattr(models, conf.model.name)
        model_params = copy.deepcopy(conf.model.model_params)
        model_params.update(
            {k: eval(param_types[k])(v) for k, v in opt_params.items()}
        )
        logger.info(f'Model Parameters:\n{pformat(model_params)}')
        model = models.LastMonthOut(
            model_class,
            model_params,
            conf.model.fit_params,
            conf.get("resample", None),
            False
        )
        
        model.fit(feature_tr, target, feats)
        
        score = model.results['trials']['Full']['val_score']
        logger.info(f'[{opt_idx:0>3}] AUROC: {score:.9f} [{(time.time() - st_time) / 60:.2f} min.]')
        result_path = os.path.join(out_dir, f'{opt_idx:0>3}_result_{score:.6f}.json')
        with open(result_path, "w") as fp:
            json.dump(model.results, fp, indent=2)        
        
        return {'loss': -1.0 * score, 'status': STATUS_OK}

    logger.info("====== optimize lgbm parameters ======")
    trials = hyperopt.Trials()
    best = fmin(objective, search_space, algo=tpe.suggest,
                max_evals=NUM_EVAL, trials=trials, verbose=1)
    logger.info("====== best estimate parameters ======")
    logger.info(pformat(best))
    logger.info("============= best score =============")
    best_score = -1.0 * trials.best_trial['result']['loss']
    logger.info(best_score)
    pickle.dump(trials.trials, open(
        os.path.join(
            out_dir,
            f'trials_score{best_score}.pkl'
        ), 'wb')
    )


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
    feature_tr, feature_te = create_feature(conf.features)

    logger.info(f"head of features\n{feature_tr.head()}\n{feature_te.head()}")

    target = read_target()

    opt(feature_tr, target, out_dir, conf)

if __name__ == "__main__":
    main()
