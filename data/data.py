import numpy as np
from pathlib import Path
from typing import Tuple
import logging

from sklearn.decomposition import PCA
from scipy.stats import zscore

class HarryPotterData():
    SKIP_WORDS = 20
    SKIP_WORDS_END = 15
#    END_WORDS = 5176
    NUM_RUNS = 4

    def __init__(self, data_directory, features_folder):
        self.dataset_name = "harry_potter"
        self.modality = "reading"
        self.all_subjects = ["F", "J", "M", "H", "K", "N", "I", "L"]
        self.data_directory = Path(data_directory)
        self.features_folder = Path(features_folder)
        self.data_by_subject = {} # each shape: (truncated num_measurements, some n_voxel value, different for participants)
        self.time_fmri = None # shape: (num_measurements,)
        self.runs_fmri = None # shape: (num_measurements,)
        self.time_words_fmri = None # shape: (num_words,)
        self.model_representations = None # shape: (layers + 1, num_words, model_hidden_dim)

        self._load_data()

        self.folds_split = None

    def _load_data(self) -> None:
        if self.time_fmri is None:
            self.time_fmri = np.load(str(self.data_directory.joinpath("time_fmri.npy"))) # (1351,) => [0, 2, 4, ...]
        if self.runs_fmri is None:
            self.runs_fmri = np.load(str(self.data_directory.joinpath("runs_fmri.npy"))) # (1351,) => [1,...,2,...,3,...,4]
        if self.time_words_fmri is None:
            self.time_words_fmri = np.load(str(self.data_directory.joinpath("time_words_fmri.npy"))) # (5176,) => [20., 20.5, ..., 2693.0]

        self.model_representations = np.load(str(self.features_folder.joinpath("representations.npy")))
        with open(self.features_folder.joinpath(f"words_skipped.json"), "r", encoding="utf-8") as file:
            words_skipped = int(file.read())

        self.time_words_fmri = self.time_words_fmri[words_skipped:]

        self.num_words = self.time_words_fmri.shape[0]
        self.num_measurements = self.runs_fmri.shape[0]
            
        self.word_measurement_num = np.zeros(self.num_words, dtype=int)
        for i in range(self.num_words):                
            self.word_measurement_num[i] = np.asarray(self.time_words_fmri[i] > self.time_fmri).nonzero()[0][-1]

        self.skip_first_trs = self.word_measurement_num.min()

        for subject in self.all_subjects:
            self.data_by_subject[subject] = np.load(
                str(self.data_directory.joinpath('data_subject_{}.npy'.format(subject)))) # (1211, n_voxels)
            if self.skip_first_trs > self.SKIP_WORDS:
                self.data_by_subject[subject] = self.data_by_subject[subject][self.skip_first_trs - self.SKIP_WORDS:, :]
        self.truncated_num_measurements = self.data_by_subject[self.all_subjects[0]].shape[0]

    def get_subject_data(self, subject):
        return self.data_by_subject[subject]
    
    def split_by_folds(self, n_folds: int, n_delays: int) -> None:
        self.folds_split = np.zeros(self.truncated_num_measurements - n_delays)
        items_in_one_fold = (self.truncated_num_measurements - n_delays) // n_folds
        for i in range(0, n_folds -1):
            self.folds_split[i * items_in_one_fold:(i + 1) * items_in_one_fold] = i
        self.folds_split[(n_folds - 1) * items_in_one_fold:] = n_folds - 1
        self.n_folds = n_folds

    def get_train_test_inds_for_fold(self, fold: int) -> Tuple[np.array, np.array]:
        assert self.folds_split is not None, "do split_by_folds() first"
        return (self.folds_split != fold), (self.folds_split == fold)

    def get_train_test_words_for_fold(self, fold, num_delays) -> Tuple[np.array, np.array]:
        train, test = self.get_train_test_inds_for_fold(fold)
        split_by_runs_sizes = [
            (self.runs_fmri == 1).sum() - max(self.skip_first_trs, self.SKIP_WORDS) - self.SKIP_WORDS_END
        ] + [
            (self.runs_fmri == i + 1).sum() - self.SKIP_WORDS - self.SKIP_WORDS_END
            for i in range(1, self.NUM_RUNS)
        ]
        train_by_runs = np.hstack([
            np.zeros(max(self.skip_first_trs, self.SKIP_WORDS) + num_delays, dtype=bool),
            train[:split_by_runs_sizes[0]],
            np.zeros(self.SKIP_WORDS_END, dtype=bool),
        ] + [
            np.hstack([
                np.zeros(self.SKIP_WORDS, dtype=bool),
                train[np.cumsum(split_by_runs_sizes)[i - 1]:np.cumsum(split_by_runs_sizes)[i]],
                np.zeros(self.SKIP_WORDS_END, dtype=bool),
            ]) for i in range(1, self.NUM_RUNS)
        ])
        test_by_runs = np.hstack([
            np.zeros(max(self.skip_first_trs, self.SKIP_WORDS) + num_delays, dtype=bool),
            test[:split_by_runs_sizes[0]],
            np.zeros(self.SKIP_WORDS_END, dtype=bool),
        ] + [
            np.hstack([
                np.zeros(self.SKIP_WORDS, dtype=bool),
                test[np.cumsum(split_by_runs_sizes)[i - 1]:np.cumsum(split_by_runs_sizes)[i]],
                np.zeros(self.SKIP_WORDS_END, dtype=bool),
            ]) for i in range(1, self.NUM_RUNS)
        ])
        words_in_train, words_in_test = np.zeros(self.num_words, dtype=bool), np.zeros(self.num_words, dtype=bool)

        # here might be bug because I do not skip first 20 words in the beginning
        for i in range(self.num_words):                
            if train_by_runs[self.word_measurement_num[i]]:
                words_in_train[i] = True
            if test_by_runs[self.word_measurement_num[i]]:
                words_in_test[i] = True
        return words_in_train, words_in_test

    def get_model_features(self, fold, num_delays, use_pca=True):
        if use_pca:
            words_in_train, words_in_test = self.get_train_test_words_for_fold(fold, num_delays)
            model_representations = np.empty((self.model_representations.shape[0], self.model_representations.shape[1], 10))
            for layer in range(self.model_representations.shape[0]):
                pca = PCA(n_components=10, svd_solver='auto')
                pca.fit(self.model_representations[layer, words_in_train, :])
                model_representations[layer, :, :] = pca.transform(self.model_representations[layer, :, :])
        else:
            model_representations = self.model_representations

        model_representations_per_tr = np.empty((model_representations.shape[0], self.num_measurements - self.skip_first_trs, model_representations.shape[-1] * num_delays))
        for i in range(self.num_measurements - self.skip_first_trs):
            model_representations_per_tr[:, i, :model_representations.shape[-1]] = np.mean(
                model_representations[:, self.word_measurement_num == i + self.skip_first_trs, :], axis=-2
            )

        for delay in range(num_delays):
            dalayed_repr = np.roll(model_representations_per_tr[:, :,:model_representations.shape[-1]], delay, axis=-2)
            dalayed_repr[:, :delay, :] = 0
            model_representations_per_tr[:, :, model_representations.shape[-1] * delay:model_representations.shape[-1] * (delay + 1)] = dalayed_repr

        model_representations_per_tr = np.hstack([
            model_representations_per_tr[:, self.runs_fmri[self.skip_first_trs:] == 1, :][:, max(self.SKIP_WORDS, self.skip_first_trs) - self.skip_first_trs:-self.SKIP_WORDS_END, :]
        ] + 
        [
            model_representations_per_tr[:, self.runs_fmri[self.skip_first_trs:] == i + 1, :][:, self.SKIP_WORDS:-self.SKIP_WORDS_END, :] for i in range(1, self.NUM_RUNS)
        ])

        return model_representations_per_tr[:, num_delays:, :]
    
    def get_brain_and_model_repr_for_regression(self, fold, subject, skip_between, num_delays, use_pca=True):
        model_representations_per_tr = self.get_model_features(fold, num_delays, use_pca)
        brain_representations = self.get_subject_data(subject)

        model_representations_per_tr = model_representations_per_tr[:, :min(model_representations_per_tr.shape[1], brain_representations.shape[0]), :]
        brain_representations = brain_representations[:min(model_representations_per_tr.shape[1], brain_representations.shape[0]), :]

        train, test = self.get_train_test_inds_for_fold(fold)
        train_brain, test_brain = brain_representations[train], brain_representations[test]
        train_model, test_model = model_representations_per_tr[:, train, :], model_representations_per_tr[:, test, :]

        # skip TRs between train and test data
        if fold == 0:
            train_brain = train_brain[skip_between: , :]
            train_model = train_model[:, skip_between:, :]
        elif fold == self.n_folds - 1:
            train_brain = train_brain[:-skip_between, :]
            train_model = train_model[:, :-skip_between, :]
        else:
            test_brain = test_brain[skip_between:-skip_between, :]
            test_model = test_model[:, skip_between:-skip_between, :]

        # normalize data
        train_brain, test_brain = zscore(train_brain, axis=0), zscore(test_brain, axis=0)
        train_model, test_model = zscore(train_model, axis=1), zscore(test_model, axis=1)

        return train_brain, test_brain, train_model, test_model
        