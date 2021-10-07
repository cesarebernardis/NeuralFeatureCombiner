import numpy as np
import scipy.sparse as sps
import tensorflow as tf
import os, datetime, random
import similaripy as sim

from deepexplain.tensorflow import DeepExplain

from RecSysFramework.Recommender import ItemSimilarityMatrixRecommender, BaseItemCBFRecommender
from RecSysFramework.Utils import EarlyStoppingModel
from RecSysFramework.Recommender.DeepLearning import NeuralFeatureCombiner


# reference: Feature Selection Using Neural Networks with Contribution Measures
# reference: Illuminating the “black box”: a randomization approach for understanding variable contributions in artificial neural networks
def input_contribution_garson(in_w, out_w, normalize=True):
    c = np.absolute(np.multiply(in_w[..., np.newaxis], out_w))
    r = c.sum(axis=0)
    w = np.divide(c, r[np.newaxis]).sum(axis=1)
    if normalize:
        w = np.divide(w, w.sum(axis=0))
    return w


# reference: An accurate comparison of methods for quantifying variable importance in artificial neural networks using simulated data
def input_contribution_connection_weights(in_w, out_w, normalize=True):
    c = np.multiply(in_w[..., np.newaxis], out_w)
    w = c.sum(axis=1)
    if normalize:
        w = np.divide(w, w.sum(axis=0))
    return w


#Methods for interpreting and understanding deep neural networks
#Learning Important Features Through Propagating Activation Differences
#A Survey of Methods for Explaining Black Box Models


class NeuralFeatureCombinerFW(BaseItemCBFRecommender, ItemSimilarityMatrixRecommender):

    RECOMMENDER_NAME = "NeuralFeatureCombinerFW"

    def __init__(self, URM_train, ICM_train, deepexplain_ctx, nfc_instance):
        super(NeuralFeatureCombinerFW, self).__init__(URM_train, ICM_train)
        self.nfc_instance = nfc_instance
        self.deepexplain_ctx = deepexplain_ctx
        self.explainer = None
        with self.nfc_instance.graph.as_default():
            self.input_tensor = tf.compat.v1.placeholder(tf.float32, [1, self.ICM_train.shape[1]])
            self.output_tensor = self.nfc_instance.feed_structure_dense(self.input_tensor)


    def fit(self, topK=200, shrink=100, similarity="cosine", importance_type="connection_weights",
            weight_per_output=False, **similarity_args):

        self.similarity = similarity
        self.topK = topK
        self.shrink = shrink
        self.importance_type = importance_type
        self.weight_per_output = weight_per_output

        self._print("{}: Generating similarity matrix".format(self.RECOMMENDER_NAME))

        weights = self.compute_weights(type=self.importance_type, per_output=self.weight_per_output)
        second_matrix = None
        print("WEIGHTS:", weights.shape)
        if self.weight_per_output:
            ICM_train = self.ICM_train
            second_matrix = sps.csr_matrix(weights.T)
        else:
            ICM_train = self.ICM_train.multiply(weights)

        if self.similarity == "cosine":
            self.W_sparse = sim.cosine(ICM_train, second_matrix, k=topK, shrink=shrink, **similarity_args)
        elif self.similarity == "jaccard":
            self.W_sparse = sim.jaccard(ICM_train, second_matrix, k=topK, shrink=shrink, **similarity_args)
        elif self.similarity == "dice":
            self.W_sparse = sim.dice(ICM_train, second_matrix, k=topK, shrink=shrink, **similarity_args)
        elif self.similarity == "tversky":
            self.W_sparse = sim.tversky(ICM_train, second_matrix, k=topK, shrink=shrink, **similarity_args)
        elif self.similarity == "splus":
            self.W_sparse = sim.s_plus(ICM_train, second_matrix, k=topK, shrink=shrink, **similarity_args)
        else:
            raise ValueError("Unknown value '{}' for similarity".format(self.similarity))

        self.W_sparse = self.W_sparse.tocsr()
        self.W_sparse.setdiag(0.0)
        self.W_sparse.eliminate_zeros()


    def compute_weights(self, features=None, type="connection_weights", per_input=True, use_output_weights=True):

        if type in ["connection_weights", "garson"]:

            enc_weights, comb_weights = self.nfc_instance.get_network_weights()

            if type == "connection_weights":
                w_function = input_contribution_connection_weights
            elif type == "garson":
                w_function = input_contribution_garson
            else:
                raise Exception("Unknown type of contribution required")

            feature_weights = None
            for i in range(len(enc_weights)-1):
                est = w_function(enc_weights[i], enc_weights[i+1], normalize=False)
                if feature_weights is None:
                    feature_weights = est
                else:
                    feature_weights = np.dot(feature_weights, est)

            comb_weights = [enc_weights[-1]] + comb_weights
            for i in range(len(comb_weights) - 1):
                est = w_function(comb_weights[i], comb_weights[i + 1], normalize=False)
                feature_weights = np.dot(feature_weights, est)
                if i == 0:
                    feature_weights = np.reshape(feature_weights,
                                                 (feature_weights.shape[0], -1, comb_weights[i].shape[1]))
            est = w_function(comb_weights[-2], comb_weights[-1].T, normalize=False)
            feature_weights = np.dot(feature_weights, est)

        else:

            # https://github.com/marcoancona/DeepExplain
            with self.nfc_instance.session.as_default() as session:
                session.run(self.nfc_instance.iterator.initializer, feed_dict={
                    self.nfc_instance.batch_size_ph: 1,
                    self.nfc_instance.train_items_ph: np.arange(1),
                })
            if features is None:
                features = self.ICM_train.tocsr().astype(np.float32)
            ys = None
            if self.explainer is None:
                self.explainer = self.deepexplain_ctx.get_explainer(type, self.output_tensor, self.input_tensor)
            feature_weights = []
            for i in range(features.shape[0]):
                if use_output_weights:
                    ys = self.nfc_instance.get_similarity(features[i]).toarray()
                feature_weights.append(self.explainer.run(features[i].toarray(), ys=ys))
            feature_weights = np.vstack(feature_weights)

        if not per_input:
            feature_weights = np.array(np.divide(sps.csr_matrix(feature_weights).sum(axis=0).flatten(), features.sum(axis=0).flatten() + 1e-12)).flatten()
            feature_weights = feature_weights / (np.absolute(feature_weights).max() + 1e-12)

        return feature_weights

