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
    def __init__(self, words_file, representations_folder, layer):
        self.feature_getter = LinguisticFeatures(words_file)
        self.representations = np.load(str(Path(representations_folder).joinpath("representations.npy")))[layer, :, :]

    def remove_feature(self, feature, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        T = zscore(self.feature_getter.get_regression_targets(feature))
        W = self.representations

        (weights, intercept), best_lambda = self.cross_val_ridge(X=T.reshape(-1, 1), Y=W, n_splits=10, lambdas=np.array([10**i for i in range(-6,10)]))
        with_removed_feature = W - np.dot(T.reshape(-1, 1), weights) - intercept

        np.save(str(save_dir.joinpath(f"removed_feature-{feature}.npy")), with_removed_feature)
        return with_removed_feature

    def regression_find_lambda(self, X, Y, X_test, Y_test, lambdas):
        error = []
        for idx, lmbda in enumerate(lambdas):
            model = Ridge(alpha=lmbda, fit_intercept=True, solver="cholesky")
            model.fit(X, Y)
            error.append(1 - r2_score(Y_test, model.predict(X_test)))
        logging.info(f"Feature Remover: errors in regression: {error}")
        return np.array(error)
    
    def regression(self, X, Y, lmbda):
        model = Ridge(alpha=lmbda, fit_intercept=True, solver="cholesky")
        model.fit(X, Y)
        print("coef shape", model.coef_.T.shape)
        return model.coef_.T, model.intercept_
    
    def cross_val_ridge(self, X, Y, n_splits, lambdas):
        errors_for_lambdas = np.zeros(lambdas.shape[0])
        print("shapes in ridge", X.shape, Y.shape)

        kf = KFold(n_splits=n_splits)
        for trn, val in kf.split(Y):
            cost = self.regression_find_lambda(
                X[trn], Y[trn], X[val], Y[val], lambdas=lambdas)
            errors_for_lambdas += cost

        best_lambda = lambdas[np.argmin(errors_for_lambdas)]
        logging.info(f"Feature Remover: beat lambda: {best_lambda}")

        weights, intercept = self.regression(X, Y, best_lambda)

        return (weights, intercept), best_lambda
