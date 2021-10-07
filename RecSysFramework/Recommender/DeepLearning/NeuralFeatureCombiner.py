import numpy as np
import scipy.sparse as sps
import tensorflow as tf
import os, datetime, random

from RecSysFramework.Recommender import ItemSimilarityMatrixRecommender, BaseItemCBFRecommender
from RecSysFramework.Utils import EarlyStoppingModel
from RecSysFramework.Recommender.DeepLearning.Utils import Dense3D, DenseSplitted, SparseToDense
from RecSysFramework.Recommender.DeepLearning.Utils import tensorflow_variable_summaries



class NeuralFeatureCombiner(BaseItemCBFRecommender, ItemSimilarityMatrixRecommender, EarlyStoppingModel):

    RECOMMENDER_NAME = "NeuralFeatureCombiner"

    def __init__(self, URM_train, ICM, S_matrix_target):

        super(NeuralFeatureCombiner, self).__init__(URM_train, ICM)
        if sps.issparse(S_matrix_target):
            self.S_matrix_target = S_matrix_target.tocsr(copy=True) / S_matrix_target.max()
        else:
            self.S_matrix_target = S_matrix_target / S_matrix_target.max()
        self.sparse_output = sps.issparse(self.S_matrix_target)
        self.graph = None
        self.session = None
        self.embedded_ICM = None


    def build_structure(self):

        with self.graph.as_default():

            self.layers = {"encoding": [], "combination": []}

            regularizer = tf.keras.regularizers.l2(self.l2_reg)

            with tf.compat.v1.name_scope("encoder_dnn"):

                last_layer_size = self.n_features
                for i, hu in enumerate(self.encoder_layers):

                    if i == 0:
                        layerclass = SparseToDense
                    else:
                        layerclass = tf.keras.layers.Dense

                    layer = layerclass(hu,
                                use_bias=True,
                                kernel_regularizer=regularizer,
                                activation=self.activation_function,
                                input_shape=(None, last_layer_size))

                    self.layers["encoding"].append(layer)
                    self.layers["encoding"].append(tf.keras.layers.BatchNormalization())
                    self.layers["encoding"].append(tf.keras.layers.Dropout(self.dropout_rate))

                    last_layer_size = hu

            n_items = len(self.train_items)
            with tf.compat.v1.name_scope("decoder_dnn"):

                for i, hu in enumerate(self.decoder_layers):

                    layertype = DenseSplitted
                    if i == 0:
                        layertype = Dense3D

                    layer = layertype(n_items, hu,
                                      use_bias=True,
                                      kernel_regularizer=regularizer,
                                      activation=self.activation_function)

                    self.layers["combination"].append(layer)
                    self.layers["combination"].append(tf.keras.layers.BatchNormalization())
                    self.layers["combination"].append(tf.keras.layers.Dropout(self.dropout_rate))

                    last_layer_size = hu

                self.last_combination = tf.compat.v1.get_variable("last_combination",
                        initializer=tf.random.truncated_normal([n_items, last_layer_size], mean=0.01, stddev=0.02),
                        regularizer=regularizer,
                        trainable=True,
                        dtype=tf.float32)

                last_bias = tf.compat.v1.get_variable("last_combination_bias",
                        initializer=tf.random.truncated_normal([n_items], mean=0.01, stddev=0.02),
                        regularizer=regularizer,
                        trainable=True,
                        dtype=tf.float32)

                self.layers["combination"].append(
                    lambda x: tf.reduce_sum(input_tensor=tf.multiply(x, self.last_combination), axis=-1) + last_bias
                )


    def feed_structure_dense(self, input, set_embedding=False):
        return self.feed_structure(tf.sparse.from_dense(input), set_embedding=set_embedding)


    def feed_structure(self, input, set_embedding=False):

        with self.graph.as_default():

            output = input

            for i, layer in enumerate(self.layers["encoding"]):
                output = layer(output)
                with tf.compat.v1.name_scope("layer_{}".format(i)):
                    tensorflow_variable_summaries(output)

            if set_embedding:
                self.embedding = output

            for i, layer in enumerate(self.layers["combination"]):
                output = layer(output)
                with tf.compat.v1.name_scope("layer_{}".format(i)):
                    tensorflow_variable_summaries(output)

        return output


    def build_model(self, items_in_batch, input, target, sample_weights):

        with self.graph.as_default():

            self.build_structure()

            batchsize = tf.size(input=items_in_batch)
            input = tf.sparse.SparseTensor(input[0], input[1],
                                           (batchsize, self.n_features))

            self.output = self.feed_structure(input, set_embedding=True)
            self.output_topK = tf.nn.top_k(self.output, self.topK_ph)

            with tf.compat.v1.name_scope("loss"):
                losses = tf.compat.v1.losses.mean_squared_error(self.output, target,
                                                                reduction=tf.compat.v1.losses.Reduction.NONE)
                losses = tf.math.reduce_sum(input_tensor=tf.math.multiply(losses, sample_weights), axis=0)
                self.loss_ph = tf.reduce_sum(
                    input_tensor=tf.math.divide(
                        losses, tf.math.reduce_sum(input_tensor=sample_weights, axis=0) + 1e-08
                    )
                )
                tensorflow_variable_summaries(self.loss_ph)

            self.summaries = tf.compat.v1.summary.merge_all()


    def fit(self, encoder_layers=None, decoder_layers=None, topK=200, epochs=30, batch_size=128, l2_reg=1e-4,
            dropout=0.2, rnd_seed=42, activation_function="relu", add_zeros_quota=0.2, graph=None, session=None,
            learning_rate=0.001, optimizer=None, optimizer_args=None, min_item_ratings=1, max_item_ratings=None,
            checkpoint_path=None, log_path=None, clear_session_after_train=True, **earlystopping_kwargs):

        if graph is None:
            self.graph = tf.Graph()
        else:
            self.graph = graph

        np.random.seed(rnd_seed)
        tf.compat.v1.random.set_random_seed(rnd_seed)

        if max_item_ratings is not None:
            assert max_item_ratings >= min_item_ratings, \
                "Max number of ratings per item must be greater or equal to min one"
        else:
            max_item_ratings = 0

        if encoder_layers is None or len(encoder_layers) <= 0:
            encoder_layers = [256]
        self.encoder_layers = encoder_layers

        if decoder_layers is None or len(decoder_layers) <= 0:
            decoder_layers = [16]
        self.decoder_layers = decoder_layers

        self.l2_reg = l2_reg
        self.activation_function = activation_function

        if checkpoint_path is not None:
            os.makedirs(checkpoint_path, exist_ok=True)
        if log_path is not None:
            os.makedirs(log_path, exist_ok=True)

        # Use only items that have a fair amount of data available (not too few and not too much)
        item_ratings = np.ediff1d(self.URM_train.tocsc().indptr)
        self.item_mask = item_ratings >= min_item_ratings
        if max_item_ratings > 0:
            self.item_mask = np.logical_and(self.item_mask, item_ratings <= max_item_ratings)
        assert self.item_mask.sum() > 0, "Not enough items with specified target constraints"
        self._print("{}: Selected {} items for training ({:.2f}\%)".format(self.RECOMMENDER_NAME, self.item_mask.sum(),
                                                                     self.item_mask.sum() / self.n_items * 100))

        self.train_items = np.arange(self.n_items, dtype=np.int32)[self.item_mask]
        self.dropout_rate = dropout
        self.batch_size = batch_size
        self.topK = topK

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
            self.topK_ph = tf.compat.v1.placeholder(tf.int32, [])
            self.train_items_ph = tf.compat.v1.placeholder(tf.int32, [None])
            n_train_items = len(self.train_items)

            self.S_stack = self.S_matrix_target.transpose()[:, self.train_items].astype(np.float32)

            if self.sparse_output:
                neg_samples = (n_train_items ** 2 - self.S_stack[self.train_items, :].data.size) * add_zeros_quota
                p_neg = int(min(n_train_items, neg_samples / n_train_items)) / n_train_items
                def gen_sample(x):
                    features = self.ICM_train[x, :].tocoo()
                    targets = self.S_stack[x, :]
                    weights = targets.astype(np.bool) + np.random.choice([0.0, 1.0],
                                                                         size=(x.size, len(self.train_items)),
                                                                         p=[1 - p_neg, p_neg]).astype(np.float32)

                    return x, np.array([features.row, features.col], dtype=np.int64).transpose(), features.data, \
                           targets.todense(), weights

            else:
                def gen_sample(x):
                    features = self.ICM_train[x, :].tocoo()
                    return x, np.array([features.row, features.col], dtype=np.int64).transpose(), features.data, \
                           self.S_stack[x, :], np.ones((x.size, len(self.train_items))).astype(np.float32)

            def tf_wrapper(x):
                return tf.numpy_function(func=gen_sample, inp=[x],
                                  Tout=[tf.int32, tf.int64, tf.float32, tf.float32, tf.float32])

            dataset = tf.data.Dataset.from_tensor_slices(self.train_items_ph)

            dataset = dataset.shuffle(buffer_size=5000)
            dataset = dataset.batch(self.batch_size_ph)
            dataset = dataset.map(tf_wrapper, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

            self.iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
            self.items_in_batch, in1, in2, self.target, self.sample_weights = self.iterator.get_next()
            self.input = (in1, in2)

            self.build_model(self.items_in_batch, self.input, self.target, self.sample_weights)

            if tf.test.is_gpu_available():
                tf_optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(tf_optimizer)

            self.training_op = tf_optimizer.minimize(self.loss_ph)


        if session is None:
            config = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=tf.data.experimental.AUTOTUNE,
                                inter_op_parallelism_threads=tf.data.experimental.AUTOTUNE)
            config.gpu_options.allow_growth = True
            self.session = tf.compat.v1.Session(config=config, graph=self.graph)
        else:
            self.session = session

        self.tf_ckpt_file = None if checkpoint_path is None else checkpoint_path + os.sep + 'tf_checkpoint'

        self._print("{}: Starting training".format(self.RECOMMENDER_NAME))

        with self.graph.as_default(), self.session.as_default() as session:

            session.run([
                tf.compat.v1.global_variables_initializer(),
                tf.compat.v1.local_variables_initializer()
            ])

            if checkpoint_path is not None:
                self.tf_saver = tf.compat.v1.train.Saver()
                latest = tf.train.latest_checkpoint(checkpoint_path)
                print("CHECKPOINT PATH", checkpoint_path)
                print("LATEST", latest)
                if latest is not None:
                    self._print("{}: Loading checkpoint".format(self.RECOMMENDER_NAME))
                    self.tf_saver.restore(session, latest)
            else:
                self.tf_saver = None

            if log_path is not None:
                self.train_writer = tf.compat.v1.summary.FileWriter(log_path, self.session.graph)
            else:
                self.train_writer = None

            self.global_step_val = session.run(self.global_step)

        self._train_with_early_stopping(epochs - self.global_step_val,
                                        algorithm_name=self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)

        self.W_sparse = self.W_sparse_best
        del self.W_sparse_best

        if clear_session_after_train:
            self.clear_session()


    def _run_epoch(self, num_epoch):

        nbatches = int(len(self.train_items) / self.batch_size) + 1
        pbar = tf.keras.utils.Progbar(nbatches, width=50, verbose=1)

        with self.graph.as_default(), self.session.as_default() as session:

            session.run(self.iterator.initializer, feed_dict={
                self.batch_size_ph: self.batch_size,
                self.train_items_ph: self.train_items,
            })
            cumloss = 0
            while True:
                try:

                    if self.train_writer is not None:
                        summaries, loss, _ = session.run(
                            [self.summaries, self.loss_ph, self.training_op]
                        )
                        self.train_writer.add_summary(summaries, num_epoch)
                    else:
                        loss, _ = session.run([self.loss_ph, self.training_op])

                    cumloss += loss
                    pbar.add(1, values=[('loss', loss)])

                except tf.errors.OutOfRangeError:
                    break

            session.run(self.increment_global_step_op)
            self.global_step_val = session.run(self.global_step)

            if self.tf_saver is not None and self.global_step_val % 5 == 0:
                self.tf_saver.save(session, self.tf_ckpt_file, global_step=self.global_step)


    def _prepare_model_for_validation(self):
        self._calculate_W_sparse(self.topK)
        return


    def _update_best_model(self):
        self.W_sparse_best = self.W_sparse.copy()
        return


    def clear_session(self):
        tf.keras.backend.clear_session()
        if self.session is not None:
            self.session.close()
        self.session = None
        self.graph = None
        self._print("------------ SESSION DELETED -----------------")


    def get_network_weights(self):
        with self.graph.as_default(), self.session.as_default() as session:
          enc_weights = []
          for layer in self.get_layers()["encoding"]:
            if isinstance(layer, tf.keras.layers.Dense):
                enc_weights.append(layer.get_weights()[0])
          comb_weights = []
          for layer in self.get_layers()["combination"][:-1]:
            if isinstance(layer, tf.keras.layers.Dense):
                comb_weights.append(layer.get_weights()[0])
          comb_weights.append(session.run(self.last_combination))
        return enc_weights, comb_weights


    def get_embedded_ICM(self):
        if self.embedded_ICM is None:
            self._calculate_embedded_ICM()
        return self.embedded_ICM


    def _calculate_embedded_ICM(self):

        self._print("{}: Generating embedded ICM matrix".format(self.RECOMMENDER_NAME))
        pbar = tf.keras.utils.Progbar(self.n_items, width=50, verbose=1)

        icm = np.zeros((self.n_items, self.encoder_layers[-1]), dtype=np.float32)
        with self.graph.as_default(), self.session.as_default() as session:
            session.run(self.iterator.initializer, feed_dict={
                self.batch_size_ph: 64,
                self.train_items_ph: np.arange(self.n_items, dtype=np.int32),
            })
            while True:
                try:
                    batch, embeddings = session.run([self.items_in_batch, self.embedding])
                    icm[batch, :] = embeddings
                    pbar.add(batch.size)
                except tf.errors.OutOfRangeError:
                    break

        self.embedded_ICM = icm


    def get_embedding(self, features):

        features = sps.csr_matrix(features).astype(np.float32)
        itc = features.shape[0]
        assert features.shape[1] == self.n_features, \
            "Wrong number of features provided ({} given, {} expected)".format(features.shape[1], self.n_features)

        self._print("{}: Generating required embeddedings".format(self.RECOMMENDER_NAME))
        pbar = tf.keras.utils.Progbar(itc, width=50, verbose=1)

        self.ICM_train = sps.vstack([self.ICM_train, features]).tocsr()
        self.S_stack = sps.vstack([self.S_stack] +
                                  ([sps.csr_matrix(([1], ([0], [0])), shape=(1, self.S_stack.shape[1]))] * itc))\
            .tocsr().astype(np.float32)

        icm = np.zeros((itc, self.encoder_layers[-1]), dtype=np.float32)
        with self.graph.as_default(), self.session.as_default() as session:
            session.run(self.iterator.initializer, feed_dict={
                self.batch_size_ph: 512,
                self.train_items_ph: np.arange(self.n_items, self.n_items + itc, dtype=np.int32),
            })
            while True:
                try:
                    batch, embeddings = session.run([self.items_in_batch, self.embedding])

                    icm[(batch - self.n_items), :] = embeddings

                    pbar.add(batch.size)
                except tf.errors.OutOfRangeError:
                    break

        self.ICM_train = self.ICM_train[:self.n_items, :].tocsr()
        self.S_stack = self.S_stack[:self.n_items, :].tocsr()

        return icm


    def get_similarity(self, features):

        features = sps.csr_matrix(features).astype(np.float32)
        itc = features.shape[0]
        topK = self.topK
        assert features.shape[1] == self.n_features, \
            "Wrong number of features provided ({} given, {} expected)".format(features.shape[1], self.n_features)

        self._print("{}: Generating required embeddedings".format(self.RECOMMENDER_NAME))
        pbar = tf.keras.utils.Progbar(itc, width=50, verbose=1)

        self.ICM_train = sps.vstack([self.ICM_train, features]).tocsr()
        self.S_stack = sps.vstack([self.S_stack] +
                                  ([sps.csr_matrix(([1], ([0], [0])), shape=(1, self.S_stack.shape[1]))] * itc))\
            .tocsr().astype(np.float32)

        indices = np.zeros((itc, topK), dtype=np.int32)
        data = np.zeros((itc, topK), dtype=np.float32)
        with self.graph.as_default(), self.session.as_default() as session:
            session.run(self.iterator.initializer, feed_dict={
                self.batch_size_ph: 512,
                self.train_items_ph: np.arange(self.n_items, self.n_items + itc, dtype=np.int32),
            })
            while True:
                try:
                    batch, (databatch, indicesbatch) = self.session.run(
                        [self.items_in_batch, self.output_topK], feed_dict={self.topK_ph: topK}
                    )
                    batch -= self.n_items
                    data[batch, :] = databatch
                    indices[batch, :] = indicesbatch
                    pbar.add(batch.size)
                except tf.errors.OutOfRangeError:
                    break

        indices = np.arange(self.n_items)[self.item_mask][indices.flatten()]

        similarity = sps.csr_matrix((data.flatten(), indices, np.arange(0, itc * topK + 1, topK)),
                                       shape=(itc, self.n_items))

        self.ICM_train = self.ICM_train[:self.n_items, :].tocsr()
        self.S_stack = self.S_stack[:self.n_items, :].tocsr()

        return similarity


    def _calculate_W_sparse(self, topK):

        self._print("{}: Generating similarity matrix".format(self.RECOMMENDER_NAME))
        pbar = tf.keras.utils.Progbar(self.n_items, width=50, verbose=1)

        indices = np.zeros((self.n_items, topK), dtype=np.int32)
        data = np.zeros((self.n_items, topK), dtype=np.float32)
        with self.session.as_default():
            self.session.run(self.iterator.initializer, feed_dict={
                self.batch_size_ph: 32,
                self.train_items_ph: np.arange(self.n_items, dtype=np.int32),
            })
            while True:
                try:

                    batch, (databatch, indicesbatch) = self.session.run(
                        [self.items_in_batch, self.output_topK], feed_dict={self.topK_ph: topK}
                    )
                    data[batch, :] = databatch
                    indices[batch, :] = indicesbatch
                    pbar.add(batch.size)
                except tf.errors.OutOfRangeError:
                    break

        indices = np.arange(self.n_items)[self.item_mask][indices.flatten()]

        self.W_sparse = sps.csr_matrix((data.flatten(), indices, np.arange(0, self.n_items * topK + 1, topK)),
                                       shape=(self.n_items, self.n_items)).transpose()
        self.W_sparse.setdiag(0.0)
        self.W_sparse.eliminate_zeros()


    def get_layers(self):
        return self.layers



class NeuralFeatureCombiner_OptimizerMask(NeuralFeatureCombiner):

    def fit(self, topK=200, epochs=30, batch_size=128, l2_reg=1e-04, dropout=0.2, rnd_seed=42,
            activation_function="relu", add_zeros_quota=0.2, learning_rate=0.001, optimizer=None, graph=None, session=None,
            min_item_ratings=1, max_item_ratings=None, checkpoint_path=None, log_path=None, evaluator_object=None,
            stop_on_validation=True, validation_every_n=None, lower_validations_allowed=5, validation_metric="RECALL",
            clear_session_after_train=True, **kwargs):

        encoder_layers = {}
        for key, value in kwargs.items():
            if "encoder_layer_" in key:
                encoder_layers[key] = value

        decoder_layers = {}
        for key, value in kwargs.items():
            if "decoder_layer_" in key:
                decoder_layers[key] = value

        optimizer_args = {}
        for key, value in kwargs.items():
            if "optimizer_args_" in key:
                optimizer_args[key.split("_")[-1]] = value

        super(NeuralFeatureCombiner_OptimizerMask, self).fit(
            encoder_layers=[int(value) for key, value in sorted(encoder_layers.items())],
            decoder_layers=[int(value) for key, value in sorted(decoder_layers.items())],
            topK=topK, epochs=epochs, batch_size=batch_size, dropout=dropout, l2_reg=l2_reg,
            rnd_seed=rnd_seed, activation_function=activation_function, add_zeros_quota=add_zeros_quota,
            learning_rate=learning_rate, optimizer=optimizer, optimizer_args=optimizer_args, graph=graph, session=session,
            min_item_ratings=min_item_ratings, max_item_ratings=max_item_ratings, checkpoint_path=checkpoint_path,
            log_path=log_path, evaluator_object=evaluator_object, validation_every_n=validation_every_n,
            lower_validations_allowed=lower_validations_allowed, validation_metric=validation_metric,
            stop_on_validation=stop_on_validation, clear_session_after_train=clear_session_after_train)

