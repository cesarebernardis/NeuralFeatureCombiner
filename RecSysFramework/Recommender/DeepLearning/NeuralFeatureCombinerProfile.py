import numpy as np
import scipy.sparse as sps
import tensorflow as tf
import os, datetime, random

from RecSysFramework.Recommender.DeepLearning import NeuralFeatureCombiner
from RecSysFramework.Recommender.DeepLearning.Utils import Dense3D, DenseSplitted, SparseToDense
from RecSysFramework.Recommender.DeepLearning.Utils import tensorflow_variable_summaries

from RecSysFramework.Utils import urm_to_coordinate_list



class NeuralFeatureCombinerProfile(NeuralFeatureCombiner):

    RECOMMENDER_NAME = "NeuralFeatureCombinerProfile"

    def __init__(self, URM_train, ICM):
        super(NeuralFeatureCombinerProfile, self).__init__(URM_train, ICM, sps.csr_matrix(([1], ([1], [1]))))


    def build_loss(self, target, prediction):
        return tf.keras.losses.MSE(target, prediction)


    def build_model(self, items_in_batch, item_input, user_input, target):

        with self.graph.as_default():

            self.build_structure()

            batchsize = tf.size(input=items_in_batch)
            n_items = len(self.train_items)

            item_input = tf.sparse.SparseTensor(item_input[0], item_input[1], (batchsize, self.n_features))

            self.output = self.feed_structure(item_input, set_embedding=True)
            self.output_topK = tf.nn.top_k(self.output, self.topK_ph)

            with tf.compat.v1.name_scope("loss"):
                user_profile = tf.sparse.SparseTensor(user_input[0], user_input[1], (batchsize, n_items))
                remove_input = tf.sparse.SparseTensor(
                    tf.transpose(tf.stack([tf.range(batchsize, dtype=tf.int64), items_in_batch])),
                    -target, (batchsize, n_items)
                )
                prediction = tf.sparse.reduce_sum(tf.sparse.add(user_profile, remove_input) * self.output, axis=-1)
                self.loss_ph = self.build_loss(target, prediction)
                tensorflow_variable_summaries(self.loss_ph)

            self.summaries = tf.compat.v1.summary.merge_all()


    def fit(self, encoder_layers=None, decoder_layers=None, topK=200, epochs=30, batch_size=128, l2_reg=1e-4,
            dropout=0.2, rnd_seed=42, activation_function="relu", add_zeros_quota=2.0,
            learning_rate=0.001, optimizer=None, optimizer_args=None, min_item_ratings=1, max_item_ratings=None,
            checkpoint_path=None, log_path=None, clear_session_after_train=True, **earlystopping_kwargs):

        self.graph = tf.Graph()

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
        self.log_path = log_path

        # Use only items that have a fair amount of data available (not too few and not too much)
        item_ratings = np.ediff1d(self.URM_train.tocsc().indptr)
        self.item_mask = item_ratings >= min_item_ratings
        if max_item_ratings > 0:
            self.item_mask = np.logical_and(self.item_mask, item_ratings <= max_item_ratings)
        assert self.item_mask.sum() > 0, "Not enough items with specified target constraints"
        self._print("{}: Selected {} items for training ({:.2f}\%)".format(self.RECOMMENDER_NAME, self.item_mask.sum(),
                                                                     self.item_mask.sum() / self.n_items * 100))

        self.train_items = np.arange(self.n_items, dtype=np.int32)[self.item_mask]
        self.add_zeros_quota = add_zeros_quota
        self.dropout_rate = dropout
        self.topK = topK
        self.batch_size = batch_size

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
            self.train_users_ph = tf.compat.v1.placeholder(tf.int32, [None])
            self.train_items_ph = tf.compat.v1.placeholder(tf.int32, [None])
            self.train_data_ph = tf.compat.v1.placeholder(tf.float32, [None])

            self.URM_train_cut = self.URM_train[:, self.train_items].tocsr().astype(np.float32)
            self.train_users_list, self.train_items_list, self.train_data = \
                urm_to_coordinate_list(self.URM_train_cut, self.add_zeros_quota, random_seed=0)
            self.train_items_mapper = np.zeros(self.n_items, dtype=np.int64)
            self.train_items_mapper[self.train_items] = np.arange(len(self.train_items), dtype=np.int64)

            def gen_sample(users, items, data):
                features = self.ICM_train[items, :].tocoo()
                profiles = self.URM_train_cut[users, :].tocoo()
                return items, self.train_items_mapper[items], \
                       np.array([features.row, features.col], dtype=np.int64).transpose(), features.data, \
                       np.array([profiles.row, profiles.col], dtype=np.int64).transpose(), profiles.data, data

            def tf_wrapper(users, items, data):
                return tf.numpy_function(func=gen_sample, inp=[users, items, data],
                            Tout=[tf.int32, tf.int64, tf.int64, tf.float32, tf.int64, tf.float32, tf.float32])

            dataset = tf.data.Dataset.from_tensor_slices((self.train_users_ph, self.train_items_ph, self.train_data_ph))

            dataset = dataset.shuffle(buffer_size=5000)
            dataset = dataset.batch(self.batch_size_ph)
            dataset = dataset.map(tf_wrapper, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

            self.iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
            self.items_in_batch, t_items, item_in1, item_in2, user_in1, user_in2, self.target = self.iterator.get_next()

            self.build_model(t_items, (item_in1, item_in2), (user_in1, user_in2), self.target)

            if tf.test.is_gpu_available():
                tf_optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(tf_optimizer)

            self.training_op = tf_optimizer.minimize(self.loss_ph)

        config = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=tf.data.experimental.AUTOTUNE,
                                inter_op_parallelism_threads=tf.data.experimental.AUTOTUNE)

        config.gpu_options.allow_growth = True

        self.tf_ckpt_file = None if checkpoint_path is None else checkpoint_path + os.sep + 'tf_checkpoint'

        self.session = tf.compat.v1.Session(config=config, graph=self.graph)
        self._print("{}: Starting training".format(self.RECOMMENDER_NAME))

        with self.graph.as_default(), self.session.as_default() as session:

            session.run([
                tf.compat.v1.global_variables_initializer(),
                tf.compat.v1.local_variables_initializer()
            ])

            if checkpoint_path is not None:
                self.tf_saver = tf.compat.v1.train.Saver()
                latest = tf.train.latest_checkpoint(checkpoint_path)
                if latest is not None:
                    self._print("{}: Loading checkpoint".format(self.RECOMMENDER_NAME))
                    self.tf_saver.restore(session, latest)
            else:
                self.tf_saver = None

            self.global_step_val = session.run(self.global_step)

        self._train_with_early_stopping(epochs - self.global_step_val,
                                        algorithm_name=self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)

        self.W_sparse = self.W_sparse_best
        del self.W_sparse_best

        #del self.URM_train_cut

        if clear_session_after_train:
            self.clear_session()


    def _run_epoch(self, num_epoch):

        nbatches = int(len(self.train_users_list) / self.batch_size) + 1
        pbar = tf.keras.utils.Progbar(nbatches, width=50, verbose=1)

        if self.log_path is not None:
            tf.profiler.experimental.start(self.log_path)

        with self.graph.as_default(), self.session.as_default() as session:

            with tf.profiler.experimental.Trace('train', step_num=num_epoch, _r=1):

                shuffle = np.random.permutation(len(self.train_users_list))

                session.run(self.iterator.initializer, feed_dict={
                    self.batch_size_ph: self.batch_size,
                    self.train_users_ph: self.train_users_list[shuffle],
                    self.train_items_ph: self.train_items[self.train_items_list[shuffle]],
                    self.train_data_ph: self.train_data[shuffle],
                })

                cumloss = 0
                while True:
                    try:
                        
                        loss, _ = session.run(
                            [self.loss_ph, self.training_op], feed_dict={self.topK_ph: self.topK}
                        )

                        cumloss += loss
                        pbar.add(1, values=[('loss', loss)])

                    except tf.errors.OutOfRangeError:
                        break

                session.run(self.increment_global_step_op)
                self.global_step_val = session.run(self.global_step)

                if self.tf_saver is not None and self.global_step_val % 5 == 0:
                    self.tf_saver.save(session, self.tf_ckpt_file, global_step=self.global_step)

                if self.global_step_val % 3 == 0:
                    self.train_users_list, self.train_items_list, self.train_data = \
                        urm_to_coordinate_list(self.URM_train_cut, self.add_zeros_quota, random_seed=self.global_step_val)

        if self.log_path is not None:
            tf.profiler.experimental.stop()


    def clear_session(self):
        tf.keras.backend.clear_session()
        if self.session is not None:
            self.session.close()
        self.session = None
        self.graph = None
        self._print("------------ SESSION DELETED -----------------")


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
                self.train_users_ph: np.ones(self.n_items, dtype=np.int32),
                self.train_data_ph: np.ones(self.n_items, dtype=np.float32),
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
        self.URM_train_cut = sps.vstack([self.URM_train_cut] +
                            ([sps.csr_matrix(([1], ([0], [0])), shape=(1, self.URM_train_cut.shape[1]))] * itc))\
                            .tocsr().astype(np.float32)
        tmp_tim = np.arange(self.n_items + itc, dtype=np.int64)
        tmp_tim[:self.n_items] = self.train_items_mapper
        self.train_items_mapper = tmp_tim

        icm = np.zeros((itc, self.encoder_layers[-1]), dtype=np.float32)
        with self.graph.as_default(), self.session.as_default() as session:
            session.run(self.iterator.initializer, feed_dict={
                self.batch_size_ph: 512,
                self.train_users_ph: np.ones(itc, dtype=np.int32),
                self.train_data_ph: np.ones(itc, dtype=np.float32),
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
        self.URM_train_cut = self.URM_train_cut[:self.n_items, :].tocsr()
        self.train_items_mapper = self.train_items_mapper[:self.n_items]

        return icm


    def _calculate_W_sparse(self, topK):

        self._print("{}: Generating similarity matrix".format(self.RECOMMENDER_NAME))
        pbar = tf.keras.utils.Progbar(self.n_items, width=50, verbose=1)

        indices = np.zeros((self.n_items, topK), dtype=np.int32)
        data = np.zeros((self.n_items, topK), dtype=np.float32)
        with self.session.as_default():
            self.session.run(self.iterator.initializer, feed_dict={
                self.batch_size_ph: 32,
                self.train_users_ph: np.ones(self.n_items, dtype=np.int32),
                self.train_data_ph: np.ones(self.n_items, dtype=np.float32),
                self.train_items_ph: np.arange(self.n_items, dtype=np.int32),
            })
            while True:
                try:

                    batch, (databatch, indicesbatch) = self.session.run([self.items_in_batch,
                                                                         self.output_topK],
                                                                        feed_dict={
                                                                            self.topK_ph: topK,
                                                                        })
                    data[batch, :] = databatch
                    indices[batch, :] = indicesbatch

                    pbar.add(batch.size)
                except tf.errors.OutOfRangeError:
                    break

        indices = self.train_items[indices.flatten()]

        self.W_sparse = sps.csc_matrix((data.flatten(), indices, np.arange(0, self.n_items * topK + 1, topK)),
                                       shape=(self.n_items, self.n_items)).tocsr()
        self.W_sparse.setdiag(0.0)
        self.W_sparse.eliminate_zeros()


    def get_layers(self):
        return self.layers



class NeuralFeatureCombinerProfileCE(NeuralFeatureCombinerProfile):

    RECOMMENDER_NAME = "NeuralFeatureCombinerProfileCE"

    def build_loss(self, target, prediction):
        return tf.keras.losses.binary_crossentropy(target, tf.math.sigmoid(prediction))



class NeuralFeatureCombinerProfile_OptimizerMask(NeuralFeatureCombinerProfile):

    def fit(self, topK=200, epochs=30, batch_size=128, l2_reg=1e-4, dropout=0.2, rnd_seed=42,
            activation_function="relu", add_zeros_quota=2.0, learning_rate=0.001, optimizer=None,
            min_item_ratings=1, max_item_ratings=None, checkpoint_path=None, log_path=None,
            evaluator_object=None, stop_on_validation=True, validation_every_n=None, lower_validations_allowed=5,
            validation_metric="RECALL", clear_session_after_train=True, **kwargs):

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

        super(NeuralFeatureCombinerProfile_OptimizerMask, self).fit(
            encoder_layers=[int(value) for key, value in sorted(encoder_layers.items())],
            decoder_layers=[int(value) for key, value in sorted(decoder_layers.items())],
            topK=topK, epochs=epochs, batch_size=batch_size, dropout=dropout, l2_reg=l2_reg,
            rnd_seed=rnd_seed, activation_function=activation_function, add_zeros_quota=add_zeros_quota,
            learning_rate=learning_rate, optimizer=optimizer, optimizer_args=optimizer_args,
            min_item_ratings=min_item_ratings, max_item_ratings=max_item_ratings, checkpoint_path=checkpoint_path,
            log_path=log_path, evaluator_object=evaluator_object, validation_every_n=validation_every_n,
            lower_validations_allowed=lower_validations_allowed, validation_metric=validation_metric,
            stop_on_validation=stop_on_validation, clear_session_after_train=clear_session_after_train)


class NeuralFeatureCombinerProfileCE_OptimizerMask(NeuralFeatureCombinerProfileCE):

    def fit(self, topK=200, epochs=30, batch_size=128, l2_reg=1e-4, dropout=0.2, rnd_seed=42,
            activation_function="relu", add_zeros_quota=2.0, learning_rate=0.001, optimizer=None,
            min_item_ratings=1, max_item_ratings=None, checkpoint_path=None, log_path=None,
            evaluator_object=None, stop_on_validation=True, validation_every_n=None, lower_validations_allowed=5,
            validation_metric="RECALL", clear_session_after_train=True, **kwargs):

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

        super(NeuralFeatureCombinerProfileCE_OptimizerMask, self).fit(
            encoder_layers=[int(value) for key, value in sorted(encoder_layers.items())],
            decoder_layers=[int(value) for key, value in sorted(decoder_layers.items())],
            topK=topK, epochs=epochs, batch_size=batch_size, dropout=dropout, l2_reg=l2_reg,
            rnd_seed=rnd_seed, activation_function=activation_function, add_zeros_quota=add_zeros_quota,
            learning_rate=learning_rate, optimizer=optimizer, optimizer_args=optimizer_args,
            min_item_ratings=min_item_ratings, max_item_ratings=max_item_ratings, checkpoint_path=checkpoint_path,
            log_path=log_path, evaluator_object=evaluator_object, validation_every_n=validation_every_n,
            lower_validations_allowed=lower_validations_allowed, validation_metric=validation_metric,
            stop_on_validation=stop_on_validation, clear_session_after_train=clear_session_after_train)
