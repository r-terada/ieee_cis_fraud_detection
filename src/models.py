import copy
import time
import logging
import sklearn
import numpy as np
import pandas as pd
import lightgbm as lgb

from tqdm import tqdm
from numba import jit
from catboost import CatBoostClassifier
from lightgbm.callback import _format_eval_result
from scipy.stats import norm
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GroupKFold
from litemort import LiteMORT

from utils import logger


class BaseModel:
    def train_and_validate():
        raise NotImplementedError

    def preict():
        raise NotImplementedError


def log_evaluation(logger, period=1, show_stdv=True, level=logging.DEBUG):
    def _callback(env):
        if period > 0 and env.evaluation_result_list and (env.iteration + 1) % period == 0:
            result = '\t'.join([_format_eval_result(x, show_stdv) for x in env.evaluation_result_list])
            logger.log(level, '[{}]\t{}'.format(env.iteration+1, result))
    _callback.order = 10
    return _callback


@jit
def fast_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += (1 - y_i)
        auc += y_i * nfalse
    auc /= (nfalse * (n - nfalse))
    return auc


def eval_auc(preds, dtrain):
    labels = dtrain.get_label()
    return 'auc', fast_auc(labels, preds), True


class LightGBM(BaseModel):

    def __init__(self, model_params, fit_params):
        self.clf = None
        self.fit_params = fit_params
        self.model_params = model_params

    def fit(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series
    ):
        st_time = time.time()
        train_dataset = lgb.Dataset(X_tr, label=y_tr)
        self.clf = lgb.train(
            self.model_params,
            train_dataset,
            **self.fit_params
        )

    def train_and_validate(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.DataFrame
    ):
        st_time = time.time()
        train_dataset = lgb.Dataset(X_tr, label=y_tr)
        val_dataset = lgb.Dataset(X_val, label=y_val)

        if "verbose_eval" in self.fit_params:
            period = self.fit_params["verbose_eval"]
        else:
            period = 100
        if "num_boost_round" not in self.fit_params:
            self.fit_params['num_boost_round'] = 100000
        self.clf = lgb.train(
            self.model_params,
            train_dataset,
            valid_sets=[train_dataset, val_dataset],
            callbacks=[log_evaluation(logger, period=period)],
            **self.fit_params
        )

        # make outputs
        oof_preds = self.clf.predict(X_val, num_iteration=self.clf.best_iteration)
        feature_importance = pd.DataFrame.from_dict({
            "feature": list(X_tr.columns),
            "importance": self.clf.feature_importance(importance_type='gain'),
        })
        result = {
            "trn_score": self.clf.best_score['training']['auc'],
            "val_score": self.clf.best_score['valid_1']['auc'],
            "best_iteration": self.clf.best_iteration,
            "elapsed_time": f'{(time.time() - st_time) / 60:.2f} min.',
            "feature_importance_top10": {
                row["feature"]: row["importance"] for i, row in feature_importance.sort_values("importance", ascending=False).head(10).iterrows()
            }
        }
        logger.info(
            f"best_iteration: {result['best_iteration']}, "
            f"train_score: {result['trn_score']:.6f}, "
            f"valid_score: {result['val_score']:.6f}"
        )
        return oof_preds, result

    def predict(self, X):
        return self.clf.predict(X, num_iteration=self.clf.best_iteration)


class LiteMort(BaseModel):
    def __init__(self, model_params, fit_params):
        self.clf = None
        self.model_params = model_params
        self.model_params.update(fit_params)

    def fit(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series
    ):
        self.clf = LiteMORT(self.model_params).fit(X_tr, y_tr)

    def train_and_validate(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.DataFrame
    ):
        st_time = time.time()

        if "num_boost_round" not in self.model_params:
            self.model_params['num_boost_round'] = 100000
        self.clf = LiteMORT(self.model_params).fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)]
        )

        # make outputs
        oof_preds = self.clf.predict_raw(X_val)
        feature_importance = pd.DataFrame.from_dict({
            "feature": list(X_tr.columns),
            "importance": self.clf.feature_importance(importance_type='gain'),
        })
        result = {
            # "trn_score": self.clf.best_score['training']['auc'],
            "val_score": roc_auc_score(y_val, oof_preds),
            "best_iteration": self.clf.best_iteration,
            "elapsed_time": f'{(time.time() - st_time) / 60:.2f} min.',
            "feature_importance_top10": {
                row["feature"]: row["importance"] for i, row in feature_importance.sort_values("importance", ascending=False).head(10).iterrows()
            }
        }
        logger.info(
            f"best_iteration: {result['best_iteration']}, "
            # f"train_score: {result['trn_score']:.6f}, "
            f"valid_score: {result['val_score']:.6f}"
        )
        return oof_preds, result

    def predict(self, X):
        return self.clf.predict_raw(X)

class CatBoost(BaseModel):

    def __init__(self, model_params, fit_params):
        self.clf = None
        self.fit_params = fit_params
        self.model_params = model_params

    def train_and_validate(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.DataFrame
    ):
        st_time = time.time()
        cat_columns = [c for c in X_tr.columns if str(X_tr[c].dtypes) in ['category', 'object']]
        self.clf = CatBoostClassifier(**self.model_params)
        self.clf.fit(
            X_tr,
            y_tr,
            cat_features=cat_columns,
            eval_set=(X_val, y_val),
            **self.fit_params
        )

        # make outputs
        trn_preds = self.clf.predict_proba(X_tr)[:, 1]
        trn_score = roc_auc_score(y_tr, trn_preds)
        oof_preds = self.clf.predict_proba(X_val)[:, 1]
        val_score = roc_auc_score(y_val, oof_preds)
        feature_importance = pd.DataFrame.from_dict({
            "feature": list(X_tr.columns),
            "importance": self.clf.get_feature_importance(),
        })
        result = {
            "trn_score": trn_score,
            "val_score": val_score,
            "best_iteration": self.clf.get_best_iteration(),
            "elapsed_time": f'{(time.time() - st_time) / 60:.2f} min.',
            "feature_importance_top10": {
                row["feature"]: row["importance"] for i, row in feature_importance.sort_values("importance", ascending=False).head(10).iterrows()
            }
        }
        logger.info(
            f"best_iteration: {result['best_iteration']}, "
            f"train_score: {result['trn_score']:.6f}, "
            f"valid_score: {result['val_score']:.6f}"
        )
        return oof_preds, result

    def predict(self, X):
        return self.clf.predict_proba(X)[:, 1]


class Blender(BaseModel):

    def __init__(self, model_params, fit_params):
        if 'method' in model_params:
            self.method = model_params['method']
        else:
            self.method = 'mean'

    def mean(self, X):
        return X.mean(axis=1)

    def rank_average(self, X):
        predictions = []
        for c in X.columns:
            predictions.append(rankdata(X[c].values) / X[c].values.shape[0])
        print(predictions)
        return sum(predictions) / len(predictions)

    def train_and_validate(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.DataFrame
    ):
        st_time = time.time()
        trn_preds = getattr(self, self.method)(X_tr)
        trn_score = roc_auc_score(y_tr, trn_preds)
        oof_preds = getattr(self, self.method)(X_val)
        val_score = roc_auc_score(y_val, oof_preds)
        result = {
            "trn_score": trn_score,
            "val_score": val_score,
            "elapsed_time": f'{(time.time() - st_time) / 60:.2f} min.',
        }
        logger.info(
            f"train_score: {result['trn_score']:.6f}, "
            f"valid_score: {result['val_score']:.6f}"
        )
        return oof_preds, result

    def predict(self, X):
        return getattr(self, self.method)(X)


class SKLearnClassifier(BaseModel):

    def __init__(self, model_params, fit_params):
        if 'scaler' in model_params:
            self.scaler = getattr(sklearn.preprocessing, model_params['scaler'])()
            del model_params['scaler']
        else:
            self.scaler = None
        self.clf = self._get_clf_class()(**model_params)
        self.fit_params = fit_params
        self.model_params = model_params

    def _get_clf_class(self):
        raise NotImplementedError

    def train_and_validate(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.DataFrame
    ):
        st_time = time.time()
        if self.scaler:
            logger.info(f"scale input with {self.scaler.__class__.__name__}")
            X_tr = self.scaler.fit_transform(X_tr.copy(), y_tr.copy())
            X_val = self.scaler.transform(X_val.copy())

        self.clf.fit(X_tr, y_tr, **self.fit_params)
        trn_preds = self.clf.predict_proba(X_tr)[:, 1]
        trn_score = roc_auc_score(y_tr, trn_preds)
        oof_preds = self.clf.predict_proba(X_val)[:, 1]
        val_score = roc_auc_score(y_val, oof_preds)
        result = {
            "trn_score": trn_score,
            "val_score": val_score,
            "elapsed_time": f'{(time.time() - st_time) / 60:.2f} min.',
        }
        logger.info(
            f"train_score: {result['trn_score']:.6f}, "
            f"valid_score: {result['val_score']:.6f}"
        )
        return oof_preds, result

    def predict(self, X):
        if self.scaler:
            X = self.scaler.transform(X.copy())
        return self.clf.predict_proba(X)[:, 1]


class LogReg(SKLearnClassifier):

    def _get_clf_class(self):
        return LogisticRegression


class RandomForest(SKLearnClassifier):

    def _get_clf_class(self):
        return RandomForestClassifier


class KNN(SKLearnClassifier):

    def _get_clf_class(self):
        return KNeighborsClassifier


class NoSplit(BaseEstimator, TransformerMixin):

    def __init__(self, model_class, model_params, fit_params, resample_conf={}):
        self.model = model_class(model_params, fit_params)
        self.resample_method = resample_conf.get("method", "no_resample")
        self.resample_params = resample_conf.get("params", {})
        self.results = {
            'resample_config': resample_conf,
            'model_params': copy.deepcopy(model_params),
            'fit_params': copy.deepcopy(fit_params)
        }

    def fit(self, X: pd.DataFrame, y: pd.Series, feats_list=None):
        if feats_list is None:
            feats_list = list(X.columns)
        self.model.fit(X[feats_list], y)

    def predict(self, X: pd.DataFrame, feats_list=None):
        if feats_list is None:
            feats_list = list(X.columns)
        return self.model.predict(X[feats_list])


class LastMonthOut(BaseEstimator, TransformerMixin):

    def __init__(self, model_class, model_params, fit_params, resample_conf={}, retrain_on_full=True):
        self.val_pred = None
        self.model_class = model_class
        self.model_params = model_params
        self.fit_params = fit_params
        self.resample_method = resample_conf.get("method", "no_resample")
        self.resample_params = resample_conf.get("params", {})
        self.retrain_on_full = retrain_on_full
        self.results = {
            'model_params': copy.deepcopy(model_params),
            'fit_params': copy.deepcopy(fit_params)
        }

    def _train_val_split(self, X: pd.DataFrame, y: pd.DataFrame):
        X['TARGET'] = y
        trn_X = X[X['DT_M'] < X['DT_M'].max()]
        trn_y = trn_X['TARGET']
        val_X = X[X['DT_M'] == X['DT_M'].max()]
        val_y = val_X['TARGET']
        del trn_X['TARGET'], val_X['TARGET']
        return trn_X, trn_y, val_X, val_y

    def fit(self, X: pd.DataFrame, y: pd.Series, feats_list=None):
        from sklearn.metrics import roc_auc_score

        if feats_list is None:
            feats_list = list(X.columns)

        X_tr, y_tr, X_val, y_val = self._train_val_split(X, y)
        X_tr, y_tr = getattr(
            Resampler, self.resample_method
        )(X_tr, y_tr, **self.resample_params)
        self.model = self.model_class(self.model_params, self.fit_params)
        self.val_pred, results = self.model.train_and_validate(X_tr[feats_list], y_tr, X_val[feats_list], y_val)
        self.results['result'] = results
        self.results['trials'] = {'Full': {'val_score': results['val_score']}}

        if self.retrain_on_full:
            logger.info('retrain model with full training data')
            if self.model_class == LightGBM:
                self.fit_params['num_boost_round'] = results['best_iteration']
                self.fit_params['early_stopping_rounds'] = None
            self.model = self.model_class(self.model_params, self.fit_params)
            self.model.fit(X[feats_list], y)

    def predict(self, X: pd.DataFrame, feats_list=None):
        if feats_list is None:
            feats_list = list(X.columns)
        return self.model.predict(X[feats_list])


class KFoldModel(BaseEstimator, TransformerMixin):

    def __init__(self, folds, model_class, model_params, fit_params, resample_conf={}, split_params={}):
        self.oof = None
        self.models = []
        self.model_class = model_class
        self.folds = folds
        self.fit_params = fit_params
        self.model_params = model_params
        self.resample_method = resample_conf.get("method", "no_resample")
        self.resample_params = resample_conf.get("params", {})
        self.split_params = split_params
        self.results = {
            'folds': str(folds),
            'model_params': copy.deepcopy(model_params),
            'fit_params': copy.deepcopy(fit_params)
        }

    def fit(self, X: pd.DataFrame, y: pd.Series, feats_list=None):
        from sklearn.metrics import roc_auc_score

        if feats_list is None:
            feats_list = list(X.columns)

        trials = {}
        self.oof = np.zeros(len(y))

        if isinstance(self.folds, GroupKFold) and 'group_key' in self.split_params:
            iterator = self.folds.split(X.values, y.values, groups=X[self.split_params['group_key']])
        else:
            iterator = self.folds.split(X.values, y.values)

        for n_fold, (trn_idx, val_idx) in enumerate(iterator):
            logger.info("fold {}".format(n_fold + 1))
            X_tr, y_tr = getattr(
                Resampler, self.resample_method
            )(X.iloc[trn_idx], y.iloc[trn_idx], **self.resample_params)
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            model = self.model_class(self.model_params, self.fit_params)
            oof_preds, results = model.train_and_validate(X_tr[feats_list], y_tr, X_val[feats_list], y_val)
            self.oof[val_idx] = oof_preds
            trials[f'Fold{n_fold + 1}'] = results
            self.models.append(model)

        score = roc_auc_score(y, self.oof)
        trials["Full"] = {
            "val_score": score,
            "val_score_mean": np.mean([trials[f"Fold{n_fold + 1}"]["val_score"] for n_fold in range(self.folds.n_splits)]),
            "val_score_std": np.std([trials[f"Fold{n_fold + 1}"]["val_score"] for n_fold in range(self.folds.n_splits)]),
        }
        self.results['trials'] = trials
        logger.info(f"CV score: {score:.7f}")

    def predict(self, X: pd.DataFrame, feats_list=None):
        if feats_list is None:
            feats_list = list(X.columns)
        predictions = np.zeros(len(X))
        for model in self.models:
            predictions += model.predict(X[feats_list]) / len(self.models)
        return predictions


class Resampler:
    @staticmethod
    def no_resample(X, y):
        return X, y

    @staticmethod
    def under_sample(X: pd.DataFrame, y: pd.Series, ratio=1.0, seed=42):
        '''
        ratio = y_1 / y_0_new
        '''
        np.random.seed(seed)
        logger.info(f'resample with under_sample: ratio={ratio}')
        n_labels = y.value_counts()
        logger.debug(f'label before sampling: ')
        logger.debug(n_labels)
        new_n_neg = int(n_labels[1] / ratio)
        assert new_n_neg <= n_labels[0], 'ratio must be >= n_pos / n_neg'
        # resample
        pos_indices = y[y == 1].index
        new_neg_indices = np.random.choice(y[y == 0].index, new_n_neg, replace=False)
        new_X = pd.concat([X.loc[pos_indices], X.loc[new_neg_indices]])
        new_y = pd.concat([y.loc[pos_indices], y.loc[new_neg_indices]])
        shuffled_indices = np.random.permutation(new_X.index)
        logger.debug(f'label after sampling: ')
        logger.debug(new_y.value_counts())
        return new_X.loc[shuffled_indices], new_y.loc[shuffled_indices]
