#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
import scipy.sparse as sps
import similaripy as sim

from RecSysFramework.Recommender import ItemSimilarityMatrixRecommender, BaseItemCBFRecommender
from RecSysFramework.Utils.FeatureWeighting import okapi_BM_25, TF_IDF


class ItemKNNCBF(BaseItemCBFRecommender, ItemSimilarityMatrixRecommender):

    """ ItemKNN recommender"""

    RECOMMENDER_NAME = "ItemKNNCBF"

    FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]


    def __init__(self, URM_train, ICM_train):
        super(ItemKNNCBF, self).__init__(URM_train, ICM_train)


    def get_embedded_ICM(self):

        ICM = self.ICM_train.astype(np.float32, copy=True)

        if self.feature_weighting_ignore_items is not None:
            ICM[self.feature_weighting_ignore_items, :] = 0.0
            ICM.eliminate_zeros()

        features_to_keep = np.arange(ICM.shape[1])[np.ediff1d(ICM.tocsc().indptr) > 0]
        ICM = ICM[:, features_to_keep]

        if self.feature_weighting == "BM25":
            iw, fw = okapi_BM_25(ICM, return_feature_weights=True)
            ICM = iw * self.ICM_train[:, features_to_keep] * fw
        elif self.feature_weighting == "TF-IDF":
            fw = TF_IDF(ICM, return_feature_weights=True)
            ICM = self.ICM_train[:, features_to_keep] * fw

        return ICM


    def get_embedding(self, features):

        ICM = self.ICM_train.astype(np.float32, copy=True)

        if self.feature_weighting_ignore_items is not None:
            ICM[self.feature_weighting_ignore_items, :] = 0.0
            ICM.eliminate_zeros()

        features_to_keep = np.arange(ICM.shape[1])[np.ediff1d(ICM.tocsc().indptr) > 0]
        ICM = ICM[:, features_to_keep]

        if self.feature_weighting == "BM25":
            iw, fw = okapi_BM_25(ICM, return_feature_weights=True)
            # I have to reproduce iw for the small matrix, but with the information in the ICM
            K1 = 1.2
            B = 0.75
            row_sums = np.ravel(features.sum(axis=1))
            length_norm = (1.0 - B) + B * row_sums / row_sums.mean()
            iw = sps.diags((K1 + 1.0) / (K1 * length_norm + 1.0))
            features = features[:, features_to_keep] * fw
        elif self.feature_weighting == "TF-IDF":
            fw = TF_IDF(ICM, return_feature_weights=True)
            features = features[:, features_to_keep] * fw

        return features


    def fit(self, topK=50, shrink=100, similarity='cosine', feature_weighting="none",
            feature_weighting_ignore_items=None, **similarity_args):

        # Similaripy returns also self similarity, which will be set to 0 afterwards
        topK += 1
        self.topK = topK
        self.shrink = shrink
        self.feature_weighting = feature_weighting
        self.feature_weighting_ignore_items = feature_weighting_ignore_items

        if feature_weighting not in self.FEATURE_WEIGHTING_VALUES:
            raise ValueError("Value for 'feature_weighting' not recognized. Acceptable values are {}, provided was '{}'"
                             .format(self.FEATURE_WEIGHTING_VALUES, feature_weighting))

        ICM = self.ICM_train.astype(np.float32, copy=True)

        if feature_weighting == "BM25":
            if feature_weighting_ignore_items is not None:
                ICM[feature_weighting_ignore_items, :] = 0.0
                ICM.eliminate_zeros()

            features_to_keep = np.arange(ICM.shape[1])[np.ediff1d(ICM.tocsc().indptr) > 0]
            ICM = ICM[:, features_to_keep]

            iw, fw = okapi_BM_25(ICM, return_feature_weights=True)
            ICM = iw * self.ICM_train[:, features_to_keep] * fw

            # ICM = okapi_BM_25(ICM)

        elif feature_weighting == "TF-IDF":
            if feature_weighting_ignore_items is not None:
                ICM[feature_weighting_ignore_items, :] = 0.0
                ICM.eliminate_zeros()

            features_to_keep = np.arange(ICM.shape[1])[np.ediff1d(ICM.tocsc().indptr) > 0]
            ICM = ICM[:, features_to_keep]

            fw = TF_IDF(ICM, return_feature_weights=True)
            ICM = self.ICM_train[:, features_to_keep] * fw

            # ICM = TF_IDF(ICM)

        if similarity == "cosine":
            self.W_sparse = sim.cosine(ICM, k=topK, shrink=shrink, **similarity_args)
        elif similarity == "jaccard":
            self.W_sparse = sim.jaccard(ICM, k=topK, shrink=shrink, **similarity_args)
        elif similarity == "dice":
            self.W_sparse = sim.dice(ICM, k=topK, shrink=shrink, **similarity_args)
        elif similarity == "tversky":
            self.W_sparse = sim.tversky(ICM, k=topK, shrink=shrink, **similarity_args)
        elif similarity == "splus":
            self.W_sparse = sim.s_plus(ICM, k=topK, shrink=shrink, **similarity_args)
        else:
            raise ValueError("Unknown value '{}' for similarity".format(similarity))

        self.W_sparse.setdiag(0)
        self.W_sparse.eliminate_zeros()
        self.W_sparse = self.W_sparse.transpose().tocsr()
