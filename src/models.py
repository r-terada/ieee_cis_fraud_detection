import copy
import time
import logging
import sklearn
import numpy as np
import pandas as pd
import lightgbm as lgb

from tqdm import tqdm
from lightgbm.callback import _format_eval_result
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

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


class LightGBM(BaseModel):

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
        train_dataset = lgb.Dataset(X_tr, label=y_tr)
        val_dataset = lgb.Dataset(X_val, label=y_val)

        if "verbose_eval" in self.fit_params:
            period = self.fit_params["verbose_eval"]
        else:
            period = 100
        num_round = 100000
        self.clf = lgb.train(
            self.model_params,
            train_dataset,
            num_round,
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


class Blender(BaseModel):

    def __init__(self, model_params, fit_params):
        pass

    def train_and_validate(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.DataFrame
    ):
        st_time = time.time()
        trn_preds = X_tr.mean(axis=1)
        trn_score = roc_auc_score(y_tr, trn_preds)
        oof_preds = X_val.mean(axis=1)
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
        return X.mean(axis=1)


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


class KFoldModel(BaseEstimator, TransformerMixin):

    def __init__(self, folds, model_class, model_params, fit_params, resample_conf={}):
        self.oof = None
        self.models = []
        self.model_class = model_class
        self.folds = folds
        self.fit_params = fit_params
        self.model_params = model_params
        self.resample_method = resample_conf.get("method", "no_resample")
        self.resample_params = resample_conf.get("params", {})
        self.results = {
            'folds': str(folds),
            'model_params': copy.deepcopy(model_params),
            'fit_params': copy.deepcopy(fit_params)
        }

    def fit(self, X: pd.DataFrame, y: pd.Series):
        from sklearn.metrics import roc_auc_score

        trials = {}
        self.oof = np.zeros(len(y))
        for n_fold, (trn_idx, val_idx) in enumerate(self.folds.split(X.values, y.values)):
            logger.info("fold {}".format(n_fold + 1))
            X_tr, y_tr = getattr(
                Resampler, self.resample_method
            )(X.iloc[trn_idx], y.iloc[trn_idx], **self.resample_params)
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            model = self.model_class(self.model_params, self.fit_params)
            oof_preds, results = model.train_and_validate(X_tr, y_tr, X_val, y_val)
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

    def predict(self, X):
        predictions = np.zeros(len(X))
        for model in self.models:
            predictions += model.predict(X) / len(self.models)
        return predictions


class TrainValSplit(BaseEstimator, TransformerMixin):

    def __init__(self, model_class, model_params, fit_params, resample_conf={}, val_size=0.2):
        self.val_size = val_size
        self.model = model_class(model_params, fit_params)
        self.resample_method = resample_conf.get("method", "no_resample")
        self.resample_params = resample_conf.get("params", {})
        self.results = {
            'val_size': val_size,
            'model_params': copy.deepcopy(model_params),
            'fit_params': copy.deepcopy(fit_params)
        }

    def fit(self, X: pd.DataFrame, y: pd.Series):
        from sklearn.metrics import roc_auc_score

        trials = {}

        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=self.val_size, random_state=42)
        X_tr, y_tr = getattr(Resampler, self.resample_method)(X_tr, y_tr, **self.resample_params)

        preds_val, results = self.model.train_and_validate(X_tr, y_tr, X_val, y_val)

        score = roc_auc_score(y_val, preds_val)
        trials["Full"] = {
            "val_score": score,
            'results': results
        }
        self.results['trials'] = trials

        logger.info(f"CV score: {results['val_score']:.7f}")

    def predict(self, X):
        return self.model.predict(X)


class TrainValSplitNoLeak(BaseEstimator, TransformerMixin):

    def __init__(self, feature_pipe, model_class, model_params, fit_params, resample_conf={}, val_size=0.2):
        self.feature_pipe = feature_pipe
        self.val_size = val_size
        self.model = model_class(model_params, fit_params)
        self.resample_method = resample_conf.get("method", "no_resample")
        self.resample_params = resample_conf.get("params", {})
        self.results = {
            'val_size': val_size,
            'model_params': copy.deepcopy(model_params),
            'fit_params': copy.deepcopy(fit_params)
        }

    def fit(self, X: pd.DataFrame, y: pd.Series):
        from sklearn.metrics import roc_auc_score

        trials = {}

        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=self.val_size, random_state=42)
        self.feature_pipe.fit(X_tr, y_tr)
        X_tr = self.feature_pipe.transform(X_tr)
        X_val = self.feature_pipe.transform(X_val)
        X_tr, y_tr = getattr(Resampler, self.resample_method)(X_tr, y_tr, **self.resample_params)

        preds_val, results = self.model.train_and_validate(X_tr, y_tr, X_val, y_val)

        score = roc_auc_score(y_val, preds_val)
        trials["Full"] = {
            "val_score": score,
            'results': results
        }
        self.results['trials'] = trials

        logger.info(f"CV score: {results['val_score']:.7f}")

    def predict(self, X):
        X = self.feature_pipe.transform(X)
        return self.model.predict(X)


class Resampler:
    @staticmethod
    def no_resample(X, y):
        return X, y

    @staticmethod
    def smote(X: pd.DataFrame, y: pd.Series, ratio=0.2, k_neighbors=5):
        from imblearn.over_sampling import SMOTE
        logger.info(f'resample with smote: ratio={ratio}, k_neighbors={k_neighbors}')
        sampler = SMOTE(
            sampling_strategy=ratio,
            k_neighbors=k_neighbors,
            random_state=1993,
            n_jobs=4
        )
        X_resampled, y_resampled = sampler.fit_sample(X.values, y.values)
        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)

    @staticmethod
    def under_sample(X: pd.DataFrame, y: pd.Series, ratio=0.2):
        from imblearn.under_sampling import RandomUnderSampler
        logger.info(f'resample with under_sample: ratio={ratio}')
        sampler = RandomUnderSampler(
            sampling_strategy=ratio,
            random_state=1993
        )
        X_resampled, y_resampled = sampler.fit_sample(X.values, y.values)
        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)

    @staticmethod
    def over_sample(X: pd.DataFrame, y: pd.Series, ratio=0.2):
        from imblearn.over_sampling import RandomOverSampler
        logger.info(f'resample with over_sample: ratio={ratio}')
        sampler = RandomOverSampler(
            sampling_strategy=ratio,
            random_state=1993
        )
        X_resampled, y_resampled = sampler.fit_sample(X.values, y.values)
        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
