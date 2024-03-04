from abc import ABC
import logging
from pathlib import Path
import json

from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from scipy.stats import zscore
import numpy as np

from alignment.model_representations.linguistic_features import LinguisticFeatures
from alignment.model_representations.model_representations import ModelRepresentations

class FeatureRemover(ABC):
    def __init__(self, words_file):
        self.feature_getter = LinguisticFeatures(words_file)

    def remove_feature(self, feature, representations_folder, layer, save_dir):
        representations = np.load(str(Path(representations_folder).joinpath("representations.npy")))[layer, :, :]
        with open(Path(representations_folder).joinpath("words_skipped.json"), "r", encoding="utf-8") as file:
            words_skipped = int(file.read())

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        T = self.feature_getter.get_regression_targets(feature)[words_skipped:]
        W = representations
        logging.info(f"Feature remover got targets with {T.shape} and representations with {W.shape}")

        (weights, intercept), best_lambda = self.cross_val_ridge(X=T.reshape(-1, 1), Y=W, n_splits=4, lambdas=np.array([10**i for i in range(-5,5)]))
        with_removed_feature = W - np.dot(T.reshape(-1, 1), weights) - intercept

        performance_before_removal = self.get_acc(X=W, Y=T, n_splits=4, n_folds=4, lambdas=np.array([10**i for i in range(-5,5)]))
        performance_after_removal = self.get_acc(X=W, Y=T, n_splits=4, n_folds=4, lambdas=np.array([10**i for i in range(-5,5)]))
        logging.info(f"Feature Remover: accuracy before: {performance_before_removal}, after: {performance_after_removal}")

        np.save(str(save_dir.joinpath("representations.npy")), with_removed_feature.reshape(-1, with_removed_feature.shape[0], with_removed_feature.shape[1]))
        with open(save_dir.joinpath("words_skipped.json"), "w", encoding="utf-8") as file:
            json.dump(words_skipped, file)
        with open(save_dir.joinpath("performance.json"), "w", encoding="utf-8") as file:
            json.dump({"before": performance_before_removal, "after": performance_after_removal}, file)
        return with_removed_feature

    def regression_find_lambda(self, X, Y, X_test, Y_test, lambdas):
        logging.info(f"Feature Remover: shapes in cross-validation, train X {X.shape}, Y {Y.shape}, test X {X_test.shape}, Y {Y_test.shape}")
        error = []
        for idx, lmbda in enumerate(lambdas):
            model = Ridge(alpha=lmbda, fit_intercept=True, solver="cholesky")
            model.fit(X, Y)
            error.append(1 - r2_score(Y_test, model.predict(X_test)))
        logging.debug(f"Feature Remover: errors in regression: {error}")
        return np.array(error)
    
    def regression(self, X, Y, lmbda):
        logging.info(f"Feature Remover: shapes in final regression X {X.shape}, Y {Y.shape}")
        model = Ridge(alpha=lmbda, fit_intercept=True, solver="cholesky")
        model.fit(X, Y)
        logging.info(f"Feature Remover: final coefs shape {model.coef_.T.shape}, {model.intercept_.shape}")
        return model.coef_.T, model.intercept_
    
    def get_acc(self, X, Y, n_splits, n_folds, lambdas):
        acc = 0
        kf = KFold(n_splits=n_folds)
        for trn, val in kf.split(Y):
            (weights, intercept), best_lambda = self.cross_val_ridge(
                X[trn], Y[trn], n_splits, lambdas)
            acc += (np.dot(X[val], weights) + intercept == Y[val]).sum()
        return acc / Y.shape[0]

    def cross_val_ridge(self, X, Y, n_splits, lambdas):
        errors_for_lambdas = np.zeros(lambdas.shape[0])

        kf = KFold(n_splits=n_splits)
        for trn, val in kf.split(Y):
            cost = self.regression_find_lambda(
                X[trn], Y[trn], X[val], Y[val], lambdas=lambdas)
            errors_for_lambdas += cost

        best_lambda = lambdas[np.argmin(errors_for_lambdas)]
        logging.info(f"Feature Remover: best lambda: {best_lambda}")

        weights, intercept = self.regression(X, Y, best_lambda)

        return (weights, intercept), best_lambda
