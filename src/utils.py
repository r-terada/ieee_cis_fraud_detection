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


def reduce_mem_usage(data, verbose=True):
    start_mem = data.memory_usage().sum() / 1024**2
    if verbose:
        print('Memory usage of dataframe: {:.2f} MB'.format(start_mem))

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    for col in data.columns:
        col_type = data[col].dtypes

        if str(col_type) in numerics:
            c_min = data[col].min()
            c_max = data[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    data[col] = data[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    data[col] = data[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    data[col] = data[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    data[col] = data[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    data[col] = data[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    data[col] = data[col].astype(np.float32)
                else:
                    data[col] = data[col].astype(np.float64)

    end_mem = data.memory_usage().sum() / 1024**2

    if verbose:
        print('Memory usage after optimization: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return data
