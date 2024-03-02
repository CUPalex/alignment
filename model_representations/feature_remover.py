from abc import ABC
import logging
from pathlib import Path

from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from scipy.stats import zscore
import numpy as np

from alignment.model_representations.linguistic_features import LinguisticFeatures
from alignment.model_representations.model_representations import ModelRepresentations

class FeatureRemover(ABC):
    def __init__(self, words_file, model_device, model_context_len, model_step, model_save_dir):
        self.model_step = model_step
        self.words = np.load(words_file)
        self.feature_getter = LinguisticFeatures(words_file)
        self.representation_getter = ModelRepresentations(model_device, model_context_len, model_step, model_save_dir)

    def remove_feature(self, feature, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        T = zscore(self.feature_getter.get_regression_targets(feature))
        W = self.representation_getter.get_model_layer_representations(self.words)

        weights, best_lambda = self.cross_val_ridge(X=T, Y=W, n_splits=10, lambdas=np.array([10**i for i in range(-6,10)]))
        with_removed_feature = W - np.dor(T, weights)

        np.save(str(save_dir.joinpath(f"removed_feature-{feature}-step-{self.model_step}.npy")), with_removed_feature)
        return with_removed_feature

    def regression_find_lambda(self, X, Y, X_test, Y_test, lambdas):
        error = []
        for idx, lmbda in enumerate(lambdas):
            model = Ridge(alpha=lmbda, fit_intercept=False, solver="cholesky")
            model.fit(X, Y)
            error.append(1 - r2_score(Y_test, model.predict(X_test)))
        logging.info(f"Feature Remover: errors in regression: {error}")
        return np.array(error)
    
    def regression(self, X, Y, lmbda):
        model = Ridge(alpha=lmbda, fit_intercept=False, solver="cholesky")
        model.fit(X, Y)
        return model.coef_.T
    
    def cross_val_ridge(self, X, Y, n_splits, lambdas):
        errors_for_lambdas = np.zeros(lambdas.shape[0])

        kf = KFold(n_splits=n_splits)
        for trn, val in kf.split(Y):
            cost = self.regression_find_lambda(
                X[trn], Y[trn], X[val], Y[val], lambdas=lambdas)
            errors_for_lambdas += cost

        best_lambda = lambdas[np.argmin(errors_for_lambdas)]
        logging.info(f"Feature Remover: beat lambda: {best_lambda}")

        weights = self.regression(X, Y, best_lambda)

        return weights, best_lambda
