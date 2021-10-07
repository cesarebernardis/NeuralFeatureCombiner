#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 08/09/17

"""

from RecSysFramework.Utils import EarlyStoppingModel
from RecSysFramework.Utils import check_matrix
from RecSysFramework.Recommender import ItemSimilarityMatrixRecommender, BaseItemCBFRecommender
from RecSysFramework.Recommender.DataIO import DataIO

import time, sys
import numpy as np
import scipy.sparse as sps
import similaripy as sim

from sklearn.preprocessing import normalize


class HP3(BaseItemCBFRecommender, ItemSimilarityMatrixRecommender, EarlyStoppingModel):

    RECOMMENDER_NAME = "HP3"

    INIT_VALUE = 1e-08

    def __init__(self, URM_train, ICM, S_matrix_target):

        super(HP3, self).__init__(URM_train, ICM)

        if URM_train.shape[1] != ICM.shape[0]:
            raise ValueError("Number of items not consistent. URM contains {} but ICM contains {}"
                             .format(URM_train.shape[1], ICM.shape[0]))

        if S_matrix_target.shape[0] != S_matrix_target.shape[1]:
            raise ValueError("Items imilarity matrix is not square: rows are {}, columns are {}"
                             .format(S_matrix_target.shape[0], S_matrix_target.shape[1]))

        if S_matrix_target.shape[0] != ICM.shape[0]:
            raise ValueError("Number of items not consistent. S_matrix contains {} but ICM contains {}"
                             .format(S_matrix_target.shape[0], ICM.shape[0]))

        self.S_matrix_target = check_matrix(S_matrix_target, 'csr')
        self.sparse_weights = True

        self.D_incremental = np.ones(self.n_features, dtype=np.float64)
        self.D_best = self.D_incremental.copy()


    def generateTrainData_low_ram(self):

        print(self.RECOMMENDER_NAME + ": Generating train data")

        start_time_batch = time.time()

        # Here is important only the structure
        self.compute_W_sparse()
        S_matrix_contentKNN = check_matrix(self.W_sparse, "csr")


        self.writeLog(self.RECOMMENDER_NAME + ": Collaborative S density: {:.2E}, nonzero cells {}".format(
            self.S_matrix_target.nnz/self.S_matrix_target.shape[0]**2, self.S_matrix_target.nnz))

        self.writeLog(self.RECOMMENDER_NAME + ": Content S density: {:.2E}, nonzero cells {}".format(
            S_matrix_contentKNN.nnz/S_matrix_contentKNN.shape[0]**2, S_matrix_contentKNN.nnz))

        num_common_coordinates = 0

        estimated_n_samples = int(S_matrix_contentKNN.nnz*(1 + self.add_zeros_quota) * 1.2)

        self.row_list = np.zeros(estimated_n_samples, dtype=np.int32)
        self.col_list = np.zeros(estimated_n_samples, dtype=np.int32)
        self.data_list = np.zeros(estimated_n_samples, dtype=np.float32)

        num_samples = 0

        for row_index in range(self.n_items):

            start_pos_content = S_matrix_contentKNN.indptr[row_index]
            end_pos_content = S_matrix_contentKNN.indptr[row_index+1]

            content_coordinates = S_matrix_contentKNN.indices[start_pos_content:end_pos_content]

            start_pos_target = self.S_matrix_target.indptr[row_index]
            end_pos_target = self.S_matrix_target.indptr[row_index+1]

            target_coordinates = self.S_matrix_target.indices[start_pos_target:end_pos_target]

            # Chech whether the content coordinate is associated to a non zero target value
            # If true, the content coordinate has a collaborative non-zero value
            # if false, the content coordinate has a collaborative zero value
            is_common = np.in1d(content_coordinates, target_coordinates)

            num_common_in_current_row = is_common.sum()
            num_common_coordinates += num_common_in_current_row

            for index in range(len(is_common)):

                if num_samples == estimated_n_samples:
                    dataBlock = 100000
                    self.row_list = np.concatenate((self.row_list, np.zeros(dataBlock, dtype=np.int32)))
                    self.col_list = np.concatenate((self.col_list, np.zeros(dataBlock, dtype=np.int32)))
                    self.data_list = np.concatenate((self.data_list, np.zeros(dataBlock, dtype=np.float32)))
                    estimated_n_samples += dataBlock

                if is_common[index]:
                    # If cell exists in target matrix, add its value
                    # Otherwise it will remain zero with a certain probability

                    col_index = content_coordinates[index]

                    self.row_list[num_samples] = row_index
                    self.col_list[num_samples] = col_index
                    self.data_list[num_samples] = self.S_matrix_target[row_index, col_index]

                    num_samples += 1

                elif np.random.rand() < self.add_zeros_quota:

                    col_index = content_coordinates[index]

                    self.row_list[num_samples] = row_index
                    self.col_list[num_samples] = col_index
                    self.data_list[num_samples] = 0.0

                    num_samples += 1

            if time.time() - start_time_batch > 30 or num_samples == S_matrix_contentKNN.nnz * (1 + self.add_zeros_quota):

                print(self.RECOMMENDER_NAME + ": Generating train data. Sample {} ( {:.2f} %) "
                      .format(num_samples, num_samples/ S_matrix_contentKNN.nnz*(1+self.add_zeros_quota) *100))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_batch = time.time()

        self.writeLog(self.RECOMMENDER_NAME + ": Content S structure has {} out of {} ( {:.2f}%) nonzero collaborative cells".format(
            num_common_coordinates, S_matrix_contentKNN.nnz, num_common_coordinates/S_matrix_contentKNN.nnz*100))

        # Discard extra cells at the left of the array
        self.row_list = self.row_list[:num_samples]
        self.col_list = self.col_list[:num_samples]
        self.data_list = self.data_list[:num_samples]

        data_nnz = sum(np.array(self.data_list)!=0)
        data_sum = sum(self.data_list)

        collaborative_nnz = self.S_matrix_target.nnz
        collaborative_sum = sum(self.S_matrix_target.data)

        self.writeLog(self.RECOMMENDER_NAME + ": Nonzero collaborative cell sum is: {:.2E}, average is: {:.2E}, "
                      "average over all collaborative data is {:.2E}".format(
                      data_sum, data_sum/data_nnz, collaborative_sum/collaborative_nnz))


    def writeLog(self, string):

        print(string)
        sys.stdout.flush()
        sys.stderr.flush()

        if self.logFile is not None:
            self.logFile.write(string + "\n")
            self.logFile.flush()


    def compute_W_sparse(self, use_incremental=False):

        if use_incremental:
            D = self.D_incremental
        else:
            D = self.D_best

        self.W_sparse = sim.dot_product(self.ICM_train * sps.diags(D), self.ICM_train.astype(np.bool).T, k=self.topK).tocsr()
        self.W_sparse = normalize(self.W_sparse, norm="l1", axis=1)
        self.sparse_weights = True


    def set_cold_start_items(self, ICM_cold):

        self.ICM_train = ICM_cold.copy()
        self.compute_W_sparse()


    def fit(self, logFile=None, precompute_common_features=True, learning_rate=1e-08, init_value=1e-08,
            use_dropout=True, dropout_perc=0.3, l1_reg=0.0, l2_reg=0.0, epochs=50, topK=300,
            add_zeros_quota=0.0, sgd_mode='adagrad', gamma=0.9, beta_1=0.9, beta_2=0.999,
            stop_on_validation=False, lower_validations_allowed=None, validation_metric="MAP",
            evaluator_object=None, validation_every_n=None):

        if init_value <= 0:
            init_value = self.INIT_VALUE
            print(self.RECOMMENDER_NAME + ": Invalid init value, using default (" + str(self.INIT_VALUE) + ")")

        # Import compiled module
        from .Cython.HP3_Similarity_Cython_SGD import HP3_Similarity_Cython_SGD

        self.logFile = logFile

        self.learning_rate = learning_rate
        self.add_zeros_quota = add_zeros_quota
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.epochs = epochs
        self.topK = topK

        self.generateTrainData_low_ram()

        weights_initialization = np.ones(self.n_features, dtype=np.float64) * init_value

        # Instantiate fast Cython implementation
        self.cythonEpoch = HP3_Similarity_Cython_SGD(self.row_list, self.col_list, self.data_list,
                                                 self.n_features, self.ICM_train, simplify_model=True,
                                                 precompute_common_features=precompute_common_features,
                                                 weights_initialization=weights_initialization,
                                                 use_dropout=use_dropout, dropout_perc=dropout_perc,
                                                 learning_rate=learning_rate, l1_reg=l1_reg, l2_reg=l2_reg,
                                                 sgd_mode=sgd_mode, gamma=gamma, beta_1=beta_1, beta_2=beta_2)

        self.D_incremental = self.cythonEpoch.get_weights()
        self.D_best = self.D_incremental.copy()

        self._train_with_early_stopping(epochs,
                                        validation_every_n=validation_every_n,
                                        stop_on_validation=stop_on_validation,
                                        validation_metric=validation_metric,
                                        lower_validations_allowed=lower_validations_allowed,
                                        evaluator_object=evaluator_object,
                                        algorithm_name=self.RECOMMENDER_NAME)

        del self.D_incremental
        self.compute_W_sparse()
        sys.stdout.flush()


    def _prepare_model_for_validation(self):
        self.D_incremental = self.cythonEpoch.get_weights()
        self.compute_W_sparse(use_incremental=True)


    def _update_best_model(self):
        self.D_best = self.D_incremental.copy()


    def _run_epoch(self, num_epoch):
       self.loss = self.cythonEpoch.epochIteration_Cython()


    def get_weights(self):
        return self.D_best


    def save_model(self, folder_path, file_name=None):

        self._print("Saving model in folder '{}'".format(self.RECOMMENDER_NAME, folder_path))

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        data_dict = {
            "D_best": self.D_best,
            "topK": self.topK,
            "sparse_weights": self.sparse_weights,
            "W_sparse": self.W_sparse
        }

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save=data_dict)

        self._print("Saving complete")
