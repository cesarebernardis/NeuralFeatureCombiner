"""
Created on 03/02/2018

"""


import similaripy as sim
import scipy.sparse as sps
import pickle, sys, time
import numpy as np

from RecSysFramework.Recommender.DataIO import DataIO
from RecSysFramework.Utils import check_matrix
from RecSysFramework.Utils import EarlyStoppingModel
from RecSysFramework.Recommender import ItemSimilarityMatrixRecommender, BaseItemCBFRecommender
from RecSysFramework.Utils.FeatureWeighting import TF_IDF, okapi_BM_25

from .Cython.FBSM_Cython_Epoch import FBSM_Cython_Epoch



class FBSM(BaseItemCBFRecommender, ItemSimilarityMatrixRecommender, EarlyStoppingModel):

    RECOMMENDER_NAME = "FBSM"

    def __init__(self, URM_train, ICM):

        super(FBSM, self).__init__(URM_train, ICM)

        self.ICM = check_matrix(ICM, 'csr')
        self.n_items_icm, self.n_features = ICM.shape


    def fit(self, topK=300, epochs=30, n_factors=2, learning_rate=1e-5, precompute_user_feature_count=False,
            l2_reg_D=0.01, l2_reg_V=0.01, sgd_mode='adam', gamma=0.9, beta_1=0.9, beta_2=0.999, init_type="zero",
            stop_on_validation=False, lower_validations_allowed=None, validation_metric="MAP",
            evaluator_object=None, validation_every_n=None):

        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.l2_reg_D = l2_reg_D
        self.l2_reg_V = l2_reg_V
        self.topK = topK
        self.epochs = epochs

        if init_type == "random":
            weights_initialization = np.random.normal(0., 0.001, self.n_features).astype(np.float64)
        elif init_type == "one":
            weights_initialization = np.ones(self.n_features, dtype=np.float64)
        elif init_type == "zero":
            weights_initialization = np.zeros(self.n_features, dtype=np.float64)
        elif init_type == "BM25":
            weights_initialization = np.ones(self.n_features, dtype=np.float64)
            self.ICM = self.ICM.astype(np.float32)
            self.ICM = okapi_BM_25(self.ICM)

        elif init_type == "TF-IDF":
            weights_initialization = np.ones(self.n_features, dtype=np.float64)
            self.ICM = self.ICM.astype(np.float32)
            self.ICM = TF_IDF(self.ICM)

        else:
            raise ValueError("FBSM: 'init_type' not recognized")

        self.cythonEpoch = FBSM_Cython_Epoch(self.URM_train, self.ICM, n_factors=self.n_factors, epochs=self.epochs,
                                                  precompute_user_feature_count=precompute_user_feature_count,
                                                  learning_rate=self.learning_rate, l2_reg_D=self.l2_reg_D,
                                                  weights_initialization=weights_initialization,
                                                  l2_reg_V=self.l2_reg_V, sgd_mode=sgd_mode, gamma=gamma,
                                                  beta_1=beta_1, beta_2=beta_2)

        self.D = self.cythonEpoch.get_D()
        self.D_best = self.D.copy()

        self.V = self.cythonEpoch.get_V()
        self.V_best = self.V.copy()

        self._train_with_early_stopping(epochs,
                                        validation_every_n=validation_every_n,
                                        stop_on_validation=stop_on_validation,
                                        validation_metric=validation_metric,
                                        lower_validations_allowed=lower_validations_allowed,
                                        evaluator_object=evaluator_object,
                                        algorithm_name=self.RECOMMENDER_NAME)

        self.D = self.D_best
        self.V = self.V_best

        self.compute_W_sparse()

        sys.stdout.flush()


    def _prepare_model_for_validation(self):
        self.D = self.cythonEpoch.get_D()
        self.V = self.cythonEpoch.get_V()
        self.compute_W_sparse()


    def _update_best_model(self):
        self.D_best = self.D.copy()
        self.V_best = self.V.copy()


    def _run_epoch(self, num_epoch):
       self.loss = self.cythonEpoch.epochIteration_Cython()


    def save_model(self, folder_path, file_name=None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict = {"D": self.D,
                     "V": self.V}

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save=data_dict)

        self._print("Saving complete")


    def load_model(self, folder_path, file_name=None):
        super(FBSM, self).load_model(folder_path, file_name=file_name)
        self.compute_W_sparse()


    def set_ICM_and_recompute_W(self, ICM_new, recompute_w=True):

        self.ICM = ICM_new.copy()

        if recompute_w:
            self.compute_W_sparse(use_D=True, use_V=True)


    def compute_W_sparse(self, use_D=True, use_V=True):

        self._print("Building similarity matrix...")

        # Diagonal
        if use_D:
            D = self.D_best
            self.W_sparse = sim.dot_product(self.ICM * sps.diags(D), self.ICM.T, k=self.topK).tocsr()
        else:
            self.W_sparse = sps.csr_matrix((self.n_items, self.n_items))

        if use_V:

            topK = self.topK + 1
            W1 = self.ICM.dot(self.V_best.T)

            data = np.empty((self.n_items, topK), dtype=np.float32)
            indices = np.empty((self.n_items, topK), dtype=np.int32)
            block_size = 400
            for start_item in range(0, self.n_items, block_size):
                end_item = min(self.n_items, start_item + block_size)
                # relevant_items_partition is block_size x cutoff
                b = W1[start_item:end_item, :].dot(W1.T)
                indices[start_item:end_item, :] = b.argpartition(-topK, axis=-1)[:, -topK:]
                data[start_item:end_item, :] = b[np.arange(end_item - start_item), indices[start_item:end_item, :].T].T
            V_weights = sps.csc_matrix((data.flatten(), indices.flatten(),
                                        np.arange(0, self.n_items * topK + 1, topK)),
                                        shape=(self.n_items, self.n_items)).tocsr()

            self.W_sparse += V_weights

        self.W_sparse.setdiag(0)
        self.W_sparse.eliminate_zeros()
        self.W_sparse.sort_indices()
