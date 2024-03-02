from abc import ABC
import logging

import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from scipy.stats import zscore

from alignment.data.data import HarryPotterData

class AlignmentComputation(ABC):
    def __init__(self) -> None:
        pass

    def run(self, data: HarryPotterData, subject: str, n_folds: int, num_delays: int):
        logging.info("Start run")
        data.split_by_folds(n_folds, num_delays)
        all_preds = {}
        corrs = {}
        
        for fold in range(n_folds):
            logging.info(f"Start fold {fold}")
            train_brain, test_brain, train_model, test_model = data.get_brain_and_model_repr_for_regression(
                fold, subject, skip_between=5, num_delays=num_delays, use_pca=True)

            for layer in range(train_model.shape[0]):
                logging.info(f"Start layer {layer} in fold {fold}")
                if fold == 0:
                    corrs[layer] = []
                    all_preds[layer] = []

                weights, chosen_lambdas = self.cross_val_ridge(
                    train_model[layer, :, :], train_brain, n_splits = 10, lambdas = np.array([10**i for i in range(-6,10)]))

                preds = np.dot(test_model[layer, :, :], weights)
                corrs[layer].append(np.mean(zscore(preds) * zscore(test_brain), axis=0))
                logging.info(f"correlations for layer {layer} fold {fold}: {corrs[layer]}")
                all_preds[layer].append(preds)
                del weights

                if fold == n_folds - 1:
                    corrs[layer] = np.hstack(corrs[layer])
                    all_preds[layer] = np.hstack(all_preds[layer])

        return corrs, all_preds
    
    def regression_find_lambda(self, X, Y, X_test, Y_test, lambdas):
        error = np.zeros((len(lambdas), Y.shape[1]))
        for idx, lmbda in enumerate(lambdas):
            model = Ridge(alpha=lmbda, fit_intercept=False, solver="cholesky")
            model.fit(X, Y)
            error[idx] = 1 - r2_score(Y_test, model.predict(X_test), multioutput="raw_values")
            logging.info(f"Alignment Computation: errors in regression for lambda idx={idx}: {error[idx]}")
        return error
    
    def regression(self, X, Y, lmbda):
        model = Ridge(alpha=lmbda, fit_intercept=False, solver="cholesky")
        model.fit(X, Y)
        return model.coef_.T
    
    def cross_val_ridge(self, X, Y, n_splits, lambdas):
        errors_for_lambdas = np.zeros((lambdas.shape[0], Y.shape[1]))

        kf = KFold(n_splits=n_splits)
        for trn, val in kf.split(Y):
            cost = self.regression_find_lambda(
                X[trn], Y[trn], X[val], Y[val], lambdas=lambdas)
            errors_for_lambdas += cost

        argmin_lambda = np.argmin(errors_for_lambdas, axis=0)
        weights = np.zeros((X.shape[1], Y.shape[1]))
        for idx_lambda in range(lambdas.shape[0]):
            logging.info(f"{idx_lambda}th lambda is the best for {(argmin_lambda == idx_lambda).sum()} voxels")
            idx_vox = (argmin_lambda == idx_lambda)
            if idx_vox.sum() > 0:
                weights[:,idx_vox] = self.regression(X, Y[:,idx_vox], lambdas[idx_lambda])

        return weights, np.array([lambdas[i] for i in argmin_lambda])
