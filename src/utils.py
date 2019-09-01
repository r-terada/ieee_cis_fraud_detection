import json
import logging
import numpy as np
import pandas as pd
from pprint import pformat
from attrdict import AttrDict
from logging import getLogger, Formatter, StreamHandler, FileHandler


logger = getLogger('main')


def init_logger(out_file, stream_log_level=logging.INFO):
    global logger
    logger.setLevel(logging.DEBUG)

    stream_handler = StreamHandler()
    stream_handler.setLevel(stream_log_level)
    stream_handler.setFormatter(Formatter('%(asctime)s [%(levelname)8s] %(message)s'))
    logger.addHandler(stream_handler)

    file_handler = FileHandler(filename=out_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(Formatter('%(asctime)s [%(levelname)8s] %(message)s'))
    logger.addHandler(file_handler)

    logger.propagate = False


def create_sub(test_df: pd.DataFrame, predictions: np.array) -> pd.DataFrame:
    sub_df = pd.DataFrame({"TransactionID": test_df["TransactionID"].values})
    sub_df["isFraud"] = predictions
    return sub_df


def create_oof(train_df: pd.DataFrame, oof_predictions: np.array) ->pd.DataFrame:
    oof_df = pd.DataFrame({"TransactionID": train_df["TransactionID"].values})
    oof_df["isFraud"] = oof_predictions
    return oof_df


def read_config(config_name: str) -> dict:
    config_file_path = f"./configs/{config_name}.json"
    with open(config_file_path, mode='r', encoding='utf-8') as fp:
        json_txt = fp.read()
        json_txt = str(json_txt).replace("'", '"').replace('True', 'true').replace('False', 'false')
        config = json.loads(json_txt)

    logger.info(pformat(config))
    return AttrDict(config)
