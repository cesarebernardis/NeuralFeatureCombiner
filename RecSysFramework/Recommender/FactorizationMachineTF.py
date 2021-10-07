import numpy as np
import scipy.sparse as sps
import subprocess
import multiprocessing
import os, glob
import tensorflow as tf

from tqdm import tqdm

from RecSysFramework.Recommender import Recommender
from RecSysFramework.Recommender import ItemSimilarityMatrixRecommender, BaseItemCBFRecommender
from RecSysFramework.Recommender.DataIO import DataIO

from RecSysFramework.Utils import EarlyStoppingModel, check_matrix, urm_to_coordinate_list



class FactorizationMachine(Recommender, EarlyStoppingModel):

    RECOMMENDER_NAME = "FactorizationMachine"


    def __init__(self, URM_train, ICM=None, UCM=None):

        super(FactorizationMachine, self).__init__(URM_train)

        self.model_file = None

        if UCM is None:
            self.UCM = sps.eye(self.n_users)
        else:
            self.UCM = UCM.copy()
        self.UCM = self.UCM.tocsr().astype(np.float32)
        self.n_user_features = self.UCM.shape[1]

        if ICM is None:
            self.ICM = sps.eye(self.n_items)
        else:
            self.ICM = ICM.copy()
        self.ICM = self.ICM.tocsr().astype(np.float32)
        self.n_item_features = self.ICM.shape[1]


    def build_structure(self):

        with self.graph.as_default():

            reg = tf.keras.regularizers.l2(self.reg)
            total_features = self.n_user_features + self.n_item_features

            self.feature_embeddings_ph = tf.compat.v1.get_variable("feature_embeddings",
                        initializer=tf.random.truncated_normal([total_features, self.rank], mean=0.0, stddev=0.1),
                        regularizer=reg, trainable=True, dtype=tf.float32)

            self.linear_weights_ph = tf.compat.v1.get_variable("linear_weights",
                        initializer=tf.random.truncated_normal([total_features], mean=0.0, stddev=0.1),
                        regularizer=reg, trainable=True, dtype=tf.float32)

            self.global_bias_ph = tf.compat.v1.get_variable("global_bias",
                        initializer=tf.random.truncated_normal([], mean=0.0, stddev=0.1),
                        regularizer=reg, trainable=True, dtype=tf.float32)


    def feed_stucture(self, input_features):

        with self.graph.as_default():

            linear = tf.sparse.reduce_sum(input_features * self.linear_weights_ph, axis=-1)

            square_input_features = tf.sparse.SparseTensor(input_features.indices,
                            tf.math.square(input_features.values), input_features.dense_shape)

            comb1 = tf.math.square(tf.sparse.sparse_dense_matmul(input_features, self.feature_embeddings_ph))
            comb2 = tf.sparse.sparse_dense_matmul(square_input_features, tf.math.square(self.feature_embeddings_ph))

            output = self.global_bias_ph + linear + 0.5 * tf.reduce_sum(comb1 - comb2, axis=-1)

        return output


    def build_model(self, items_in_batch, features_input, target):

        with self.graph.as_default():

            self.build_structure()

            batchsize = tf.size(input=items_in_batch)
            total_features = self.n_user_features + self.n_item_features

            features_input = tf.sparse.SparseTensor(features_input[0], features_input[1], (batchsize, total_features))
            self.output = self.feed_structure(features_input)

            with tf.compat.v1.name_scope("loss"):
                if self.problem_type == "regression":
                    self.loss_ph = tf.keras.losses.MSE(target, self.output)
                elif self.problem_type == "classification":
                    self.loss_ph = -tf.reduce_sum(tf.math.log_sigmoid(tf.multiply(target, self.output)))
                else:
                    raise Exception("Invalid problem_type provided")
                tensorflow_variable_summaries(self.loss_ph)

            self.summaries = tf.compat.v1.summary.merge_all()


    def _compute_item_score(self, user_id_array, items_to_compute=None):

        if not isinstance(user_id_array, np.ndarray):
            user_id_array = np.array(user_id_array)

        if items_to_compute is None:
            items_to_compute = np.arange(self.n_items, dtype=np.int32)

        predictions = -np.inf * np.one((user_id_array.size, self.n_items), dtype=np.float32)

        self._print("{}: Generating predictions".format(self.RECOMMENDER_NAME))
        pbar = tf.keras.utils.Progbar(user_id_array.size, width=50, verbose=1)

        for i, user in enumerate(user_id_array):

            with self.session.as_default():
                self.session.run(self.iterator.initializer, feed_dict={
                    self.batch_size_ph: 1024,
                    self.train_users_ph: np.repeat(user, len(items_to_compute), dtype=np.int32),
                    self.train_items_ph: items_to_compute.astype(np.int32),
                })
                while True:
                    try:
                        iib, output = self.session.run([self.items_in_batch, self.output])
                        predictions[i, :][iib] = output.flatten()
                    except tf.errors.OutOfRangeError:
                        break

            pbar.add(1)

        return predictions


    def fit(self, rank=10, epochs=30, reg=1e-4, learning_rate=0.001, add_zeros_quota=2.0, batch_size=32,
            problem_type="regression", optimizer=None, optimizer_args=None, rnd_seed=42, **earlystopping_kwargs):

        self.graph = tf.Graph()

        np.random.seed(rnd_seed)
        tf.compat.v1.random.set_random_seed(rnd_seed)

        self.problem_type = problem_type
        self.add_zeros_quota = add_zeros_quota
        self.rank = rank
        self.reg = reg
        self.learning_rate = learning_rate
        self.optimizer = optimizer

        with self.graph.as_default():

            if optimizer is None:
                optimizer = "adam"

            if optimizer_args is None:
                optimizer_args = {}

            if optimizer == "adam":
                tf_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, **optimizer_args)
            elif optimizer == "adagrad":
                tf_optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate=learning_rate, **optimizer_args)
            elif optimizer == "adadelta":
                tf_optimizer = tf.compat.v1.train.AdadeltaOptimizer(learning_rate=learning_rate, **optimizer_args)
            elif optimizer == "rmsprop":
                tf_optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=learning_rate, **optimizer_args)
            elif optimizer == "sgd":
                tf_optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate, **optimizer_args)
            else:
                raise ValueError("{}: Unknown value of optimizer".format(self.RECOMMENDER_NAME))

            self.global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int32)
            self.increment_global_step_op = tf.compat.v1.assign(self.global_step, self.global_step + 1)
            self.batch_size_ph = tf.compat.v1.placeholder(tf.int64, [])
            self.train_users_ph = tf.compat.v1.placeholder(tf.int32, [None])
            self.train_items_ph = tf.compat.v1.placeholder(tf.int32, [None])
            self.train_data_ph = tf.compat.v1.placeholder(tf.float32, [None])

            def gen_sample(users, items, data):
                features = sps.vstack([self.UCM[users, :], self.ICM[items, :]]).tocoo()
                return items, np.array([features.row, features.col], dtype=np.int64).transpose(), features.data, data

            def tf_wrapper(users, items, data):
                return tf.numpy_function(func=gen_sample, inp=[users, items, data],
                                  Tout=[tf.int32, tf.int64, tf.float32, tf.float32])

            dataset = tf.data.Dataset.from_tensor_slices(
                (self.train_users_ph, self.train_items_ph, self.train_data_ph)
            )

            dataset = dataset.shuffle(buffer_size=5000)
            dataset = dataset.batch(self.batch_size_ph)
            dataset = dataset.map(tf_wrapper, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

            self.iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
            self.items_in_batch, f_in1, f_in2, self.target = self.iterator.get_next()

            self.build_model(self.items_in_batch, (f_in1, f_in2), self.target)

            if tf.test.is_gpu_available():
                tf_optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(tf_optimizer)

            self.training_op = tf_optimizer.minimize(self.loss_ph)

        config = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                          intra_op_parallelism_threads=tf.data.experimental.AUTOTUNE,
                                          inter_op_parallelism_threads=tf.data.experimental.AUTOTUNE)

        config.gpu_options.allow_growth = True

        self.session = tf.compat.v1.Session(config=config, graph=self.graph)
        self._print("{}: Starting training".format(self.RECOMMENDER_NAME))

        with self.graph.as_default(), self.session.as_default() as session:

            session.run([
                tf.compat.v1.global_variables_initializer(),
                tf.compat.v1.local_variables_initializer()
            ])

            self.global_step_val = session.run(self.global_step)

        self._train_with_early_stopping(epochs - self.global_step_val,
                                        algorithm_name=self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)

        if clear_session_after_train:
            self.clear_session()


    def _run_epoch(self, num_epoch):

        if num_epoch % 5 == 0:
            self.train_users, self.train_items, self.train_data = \
                urm_to_coordinate_list(self.URM_train, self.add_zeros_quota, random_seed=num_epoch)

        nbatches = int(len(self.train_items) / self.batch_size) + 1
        pbar = tf.keras.utils.Progbar(nbatches, width=50, verbose=1)

        with self.graph.as_default(), self.session.as_default() as session:

            session.run(self.iterator.initializer, feed_dict={
                self.batch_size_ph: self.batch_size,
                self.train_users_ph: self.train_users,
                self.train_items_ph: self.train_items,
                self.train_data_ph: self.train_data,
            })

            cumloss = 0
            while True:
                try:
                    loss, _ = session.run([self.loss_ph, self.training_op])
                    cumloss += loss
                    pbar.add(1, values=[('loss', loss)])
                except tf.errors.OutOfRangeError:
                    break

            session.run(self.increment_global_step_op)
            self.global_step_val = session.run(self.global_step)


    def clear_session(self):
        tf.keras.backend.clear_session()
        if self.session is not None:
            self.session.close()
        self.session = None
        self.graph = None
        self._print("------------ SESSION DELETED -----------------")


    def _prepare_model_for_validation(self):
        #Nothing to do
        return


    def _update_best_model(self):
        #Nothing to do
        return



class FactorizationMachineProfile(FactorizationMachine):

    RECOMMENDER_NAME = "FactorizationMachineProfile"

    def __init__(self, URM_train, ICM=None):
        super(FactorizationMachineProfile, self).__init__(URM_train, ICM=ICM, UCM=URM_train)



class FactorizationMachineSimilarity(FactorizationMachine, ItemSimilarityMatrixRecommender, BaseItemCBFRecommender):

    RECOMMENDER_NAME = "FactorizationMachineSimilarity"

    def __init__(self, URM_train, ICM, S_matrix=None):
        super(FactorizationMachineSimilarity, self).__init__(URM_train, ICM=ICM, UCM=None)
        self.S_matrix = check_matrix(S_matrix, 'csr')
