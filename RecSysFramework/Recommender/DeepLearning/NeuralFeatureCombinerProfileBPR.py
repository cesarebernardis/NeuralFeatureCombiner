import numpy as np
import scipy.sparse as sps
import tensorflow as tf
import os, datetime, random

from RecSysFramework.Recommender.DeepLearning import NeuralFeatureCombiner
from RecSysFramework.Recommender.DeepLearning.Utils import Dense3D, DenseSplitted, SparseToDense
from RecSysFramework.Recommender.DeepLearning.Utils import tensorflow_variable_summaries



class NeuralFeatureCombinerProfileBPR(NeuralFeatureCombiner):

    RECOMMENDER_NAME = "NeuralFeatureCombinerProfileBPR"

    def __init__(self, URM_train, ICM):
        super(NeuralFeatureCombinerProfileBPR, self).__init__(URM_train, ICM, sps.csr_matrix(([1], ([1], [1]))))


    def build_model(self, items_in_batch, item_pos_input, item_neg_input, user_input):

        with self.graph.as_default():

            self.build_structure()

            n_items = len(self.train_items)

            batchsize = tf.size(input=items_in_batch)
            item_pos_input = tf.sparse.SparseTensor(item_pos_input[0], item_pos_input[1],
                                                    (batchsize, self.n_features))
            self.output = self.feed_structure(item_pos_input, set_embedding=True)
            self.output_topK = tf.nn.top_k(self.output, self.topK_ph)

            with tf.compat.v1.name_scope("loss"):

                item_neg_input = tf.sparse.SparseTensor(item_neg_input[0], item_neg_input[1],
                                                        (batchsize, self.n_features))
                output_neg = self.feed_structure(item_neg_input, set_embedding=False)

                user_profile = tf.sparse.SparseTensor(user_input[0], user_input[1], (batchsize, n_items))
                remove_input_pos = tf.sparse.SparseTensor(
                    tf.transpose(tf.stack([tf.range(batchsize, dtype=tf.int64), items_in_batch])),
                    -tf.ones(batchsize, dtype=tf.float32), (batchsize, n_items)
                )

                xui = tf.sparse.reduce_sum(tf.sparse.add(user_profile, remove_input_pos) * self.output, axis=-1)
                xuj = tf.sparse.reduce_sum(user_profile * output_neg, axis=-1)
                self.loss_ph = -tf.math.reduce_sum(tf.math.log_sigmoid(xui - xuj))
                tensorflow_variable_summaries(self.loss_ph)

            self.summaries = tf.compat.v1.summary.merge_all()


    def generate_negatives(self, pop_alpha=0., random_seed=42):
        if random_seed is not None:
            np.random.seed(random_seed)
        popularity = np.power(np.ediff1d(self.URM_train_cut.tocsc().indptr), pop_alpha)
        totpop = popularity.sum()
        users, positives, negatives = [], [], []
        for user in range(self.n_users):
            user_positives = self.URM_train_cut.indices[self.URM_train_cut.indptr[user]:self.URM_train_cut.indptr[user+1]]
            backup = popularity[user_positives]
            popularity[user_positives] = 0.
            users.append(np.repeat(user, len(user_positives)))
            positives.append(user_positives)
            negatives.append(np.random.choice(self.URM_train_cut.shape[1], len(user_positives), replace=True,
                                              p=popularity / (totpop - backup.sum())))
            popularity[user_positives] = backup
        return np.concatenate(users, axis=None).astype(np.int32), \
               np.concatenate(positives, axis=None).astype(np.int32), \
               np.concatenate(negatives, axis=None).astype(np.int32)


    def fit(self, encoder_layers=None, decoder_layers=None, topK=200, epochs=30, batch_size=128, l2_reg=1e-4,
            dropout=0.2, rnd_seed=42, activation_function="relu", bpr_pop_alpha=0.,
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

        self.activation_function = activation_function
        self.l2_reg = l2_reg

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
        self.bpr_pop_alpha = bpr_pop_alpha
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
            self.train_items_pos_ph = tf.compat.v1.placeholder(tf.int32, [None])
            self.train_items_neg_ph = tf.compat.v1.placeholder(tf.int32, [None])

            self.URM_train_cut = self.URM_train[:, self.train_items].tocsr().astype(np.float32)
            self.train_items_mapper = np.zeros(self.n_items, dtype=np.int64)
            self.train_items_mapper[self.train_items] = np.arange(len(self.train_items), dtype=np.int64)

            def gen_sample(users, items, negatives):
                features_pos = self.ICM_train[items, :].tocoo()
                features_neg = self.ICM_train[negatives, :].tocoo()
                profiles = self.URM_train_cut[users, :].tocoo()
                return items, self.train_items_mapper[items], self.train_items_mapper[negatives], \
                       np.array([features_pos.row, features_pos.col], dtype=np.int64).transpose(), features_pos.data, \
                       np.array([features_neg.row, features_neg.col], dtype=np.int64).transpose(), features_neg.data, \
                       np.array([profiles.row, profiles.col], dtype=np.int64).transpose(), profiles.data

            def tf_wrapper(users, items_pos, items_neg):
                return tf.numpy_function(func=gen_sample, inp=[users, items_pos, items_neg],
                                         Tout=[
                                            tf.int32, tf.int64, tf.int64,
                                            tf.int64, tf.float32,
                                            tf.int64, tf.float32,
                                            tf.int64, tf.float32
                ])

            dataset = tf.data.Dataset.from_tensor_slices((self.train_users_ph, self.train_items_pos_ph,
                                                         self.train_items_neg_ph))

            dataset = dataset.shuffle(buffer_size=5000)
            dataset = dataset.batch(self.batch_size_ph)
            dataset = dataset.map(tf_wrapper, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

            self.iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
            self.items_in_batch, pos_items, neg_items, item_pos_in1, item_pos_in2, \
            item_neg_in1, item_neg_in2, user_in1, user_in2 = self.iterator.get_next()

            self.build_model(pos_items, (item_pos_in1, item_pos_in2),
                             (item_neg_in1, item_neg_in2), (user_in1, user_in2))

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

        #del self.URM_train_cut

        if clear_session_after_train:
            self.clear_session()


    def _run_epoch(self, num_epoch):

        feed_dict = self._get_feed_dictionary(only_items=False,
                        recalculate_input=(num_epoch % 3 == 0), random_seed=num_epoch)

        nbatches = int(len(self.URM_train_cut.data) / self.batch_size) + 1
        pbar = tf.keras.utils.Progbar(nbatches, width=50, verbose=1)

        with self.graph.as_default(), self.session.as_default() as session:

            session.run(self.iterator.initializer, feed_dict=feed_dict)

            cumloss = 0
            while True:
                try:

                    if self.train_writer is not None:
                        summaries, loss, _ = session.run(
                            [self.summaries, self.loss_ph, self.training_op], feed_dict={self.topK_ph: self.topK}
                        )
                        self.train_writer.add_summary(summaries, num_epoch)
                    else:
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
            session.run(self.iterator.initializer,
                feed_dict=self._get_feed_dictionary(only_items=True, recalculate_input=False)
            )
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
        tmp_tim = np.arange(self.n_items + itc, dtype=np.int64)
        tmp_tim[:self.n_items] = self.train_items_mapper
        self.train_items_mapper = tmp_tim

        icm = np.zeros((itc, self.encoder_layers[-1]), dtype=np.float32)
        with self.graph.as_default(), self.session.as_default() as session:
            feed_dict = {
                self.batch_size_ph: 32,
                self.train_items_pos_ph: np.arange(self.n_items, self.n_items + itc, dtype=np.int32),
                self.train_items_neg_ph: np.ones(itc, dtype=np.int32),
                self.train_users_ph: np.ones(itc, dtype=np.int32),
            }
            session.run(self.iterator.initializer, feed_dict=feed_dict)
            while True:
                try:
                    batch, embeddings = session.run([self.items_in_batch, self.embedding])

                    icm[(batch - self.n_items), :] = embeddings

                    pbar.add(batch.size)
                except tf.errors.OutOfRangeError:
                    break

        self.ICM_train = self.ICM_train[:self.n_items, :].tocsr()
        self.train_items_mapper = self.train_items_mapper[:self.n_items]

        return icm


    def _calculate_W_sparse(self, topK):

        self._print("{}: Generating similarity matrix".format(self.RECOMMENDER_NAME))
        pbar = tf.keras.utils.Progbar(self.n_items, width=50, verbose=1)

        indices = np.zeros((self.n_items, topK), dtype=np.int32)
        data = np.zeros((self.n_items, topK), dtype=np.float32)
        with self.session.as_default():
            self.session.run(self.iterator.initializer,
                feed_dict=self._get_feed_dictionary(only_items=True, recalculate_input=False)
            )
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


    def _get_feed_dictionary(self, only_items=False, recalculate_input=False, random_seed=42):

        if recalculate_input:
            self.train_users, self.train_positives, self.train_negatives = \
                self.generate_negatives(pop_alpha=self.bpr_pop_alpha, random_seed=random_seed)

        if only_items:
            feed_dict = {
                self.batch_size_ph: 32,
                self.train_items_pos_ph: np.arange(self.n_items, dtype=np.int32),
                self.train_items_neg_ph: np.ones(self.n_items, dtype=np.int32),
                self.train_users_ph: np.ones(self.n_items, dtype=np.int32),
            }
        else:
            feed_dict = {
                self.batch_size_ph: self.batch_size,
                self.train_users_ph: self.train_users,
                self.train_items_pos_ph: self.train_items[self.train_positives],
                self.train_items_neg_ph: self.train_items[self.train_negatives],
            }

        return feed_dict



class NeuralFeatureCombinerProfileBPRMax(NeuralFeatureCombinerProfileBPR):

    RECOMMENDER_NAME = "NeuralFeatureCombinerProfileBPRMax"

    def __init__(self, URM_train, ICM):
        super(NeuralFeatureCombinerProfileBPRMax, self).__init__(URM_train, ICM)


    def build_model(self, items_in_batch, item_pos_input, item_neg_input, user_input):

        with self.graph.as_default():

            self.build_structure()

            n_items = len(self.train_items)

            batchsize = tf.size(input=items_in_batch)
            item_pos_input = tf.sparse.SparseTensor(item_pos_input[0], item_pos_input[1],
                                                    (batchsize, self.n_features))
            self.output = self.feed_structure(item_pos_input, set_embedding=True)
            self.output_topK = tf.nn.top_k(self.output, self.topK_ph)

            with tf.compat.v1.name_scope("loss"):

                item_neg_input = tf.sparse.SparseTensor(item_neg_input[0], item_neg_input[1],
                                                        (batchsize * self.neg_samples, self.n_features))

                user_profile = tf.sparse.SparseTensor(user_input[0], user_input[1], (batchsize, n_items))
                remove_input_pos = tf.sparse.SparseTensor(
                    tf.transpose(tf.stack([tf.range(batchsize, dtype=tf.int64), items_in_batch])),
                    -tf.ones(batchsize, dtype=tf.float32), (batchsize, n_items)
                )

                xui = tf.sparse.reduce_sum(tf.sparse.add(user_profile, remove_input_pos) * self.output, axis=-1)

                output_neg = self.feed_structure(item_neg_input, set_embedding=False)

                user_neg_profile = tf.sparse.SparseTensor(
                    tf.gather(tf.concat([
                        tf.repeat(user_input[0], self.neg_samples, axis=0),
                        tf.reshape(
                            tf.tile(tf.range(self.neg_samples, dtype=tf.int64), [tf.size(input=user_input[1])]),
                            [-1, 1]
                        )], axis=1), [0,2,1], axis=1),
                    tf.repeat(user_input[1], self.neg_samples),
                    (batchsize, self.neg_samples, n_items)
                )
                xuj = tf.sparse.reduce_sum(
                        user_neg_profile * tf.reshape(output_neg, [batchsize, self.neg_samples, n_items]), axis=-1
                )

                weights = tf.nn.softmax(xuj, axis=-1)
                self.loss_ph = -tf.math.reduce_sum(
                    tf.math.log(
                        tf.math.reduce_sum(weights * tf.math.sigmoid(tf.reshape(xui, [-1, 1]) - xuj), axis=-1)
                    )
                )
                tensorflow_variable_summaries(self.loss_ph)

            self.summaries = tf.compat.v1.summary.merge_all()


    def generate_negatives(self, pop_alpha=0., random_seed=42):
        if random_seed is not None:
            np.random.seed(random_seed)
        popularity = np.power(np.ediff1d(self.URM_train_cut.tocsc().indptr), pop_alpha)
        totpop = popularity.sum()
        users, positives, negatives = [], [], []
        for user in range(self.n_users):
            user_positives = self.URM_train_cut.indices[self.URM_train_cut.indptr[user]:self.URM_train_cut.indptr[user+1]]
            backup = popularity[user_positives]
            popularity[user_positives] = 0.
            users.append(np.repeat(user, len(user_positives)))
            positives.append(user_positives)
            negatives.append(np.random.choice(self.URM_train_cut.shape[1], self.neg_samples * len(user_positives),
                        replace=True, p=popularity / (totpop - backup.sum())))
            popularity[user_positives] = backup
        return np.concatenate(users, axis=None).astype(np.int32), \
               np.concatenate(positives, axis=None).astype(np.int32), \
               np.concatenate(negatives, axis=None).astype(np.int32).reshape((-1, self.neg_samples))


    def fit(self, encoder_layers=None, decoder_layers=None, topK=200, epochs=30, batch_size=128, l2_reg=1e-4,
            dropout=0.2, rnd_seed=42, activation_function="relu", bpr_pop_alpha=0., neg_samples=5,
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

        self.activation_function = activation_function
        self.l2_reg = l2_reg

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
        self.bpr_pop_alpha = bpr_pop_alpha
        self.dropout_rate = dropout
        self.topK = topK
        self.batch_size = batch_size
        self.neg_samples = neg_samples

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
            self.train_items_pos_ph = tf.compat.v1.placeholder(tf.int32, [None])
            self.train_items_neg_ph = tf.compat.v1.placeholder(tf.int32, [None, self.neg_samples])

            self.URM_train_cut = self.URM_train[:, self.train_items].tocsr().astype(np.float32)
            self.train_items_mapper = np.zeros(self.n_items, dtype=np.int64)
            self.train_items_mapper[self.train_items] = np.arange(len(self.train_items), dtype=np.int64)

            def gen_sample(users, items, negatives):
                negatives = negatives.flatten()
                features_pos = self.ICM_train[items, :].tocoo()
                features_neg = self.ICM_train[negatives, :].tocoo()
                profiles = self.URM_train_cut[users, :].tocoo()
                return items, self.train_items_mapper[items], self.train_items_mapper[negatives], \
                       np.array([features_pos.row, features_pos.col], dtype=np.int64).transpose(), features_pos.data, \
                       np.array([features_neg.row, features_neg.col], dtype=np.int64).transpose(), features_neg.data, \
                       np.array([profiles.row, profiles.col], dtype=np.int64).transpose(), profiles.data

            def tf_wrapper(users, items_pos, items_neg):
                return tf.numpy_function(func=gen_sample, inp=[users, items_pos, items_neg],
                                         Tout=[
                                            tf.int32, tf.int64, tf.int64,
                                            tf.int64, tf.float32,
                                            tf.int64, tf.float32,
                                            tf.int64, tf.float32
                ])

            dataset = tf.data.Dataset.from_tensor_slices((self.train_users_ph, self.train_items_pos_ph,
                                                         self.train_items_neg_ph))

            dataset = dataset.shuffle(buffer_size=5000)
            dataset = dataset.batch(self.batch_size_ph)
            dataset = dataset.map(tf_wrapper, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

            self.iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
            self.items_in_batch, pos_items, neg_items, item_pos_in1, item_pos_in2, \
            item_neg_in1, item_neg_in2, user_in1, user_in2 = self.iterator.get_next()

            self.build_model(pos_items, (item_pos_in1, item_pos_in2),
                             (item_neg_in1, item_neg_in2), (user_in1, user_in2))

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

        #del self.URM_train_cut

        if clear_session_after_train:
            self.clear_session()


    def _get_feed_dictionary(self, only_items=False, recalculate_input=False, random_seed=42):

        if recalculate_input:
            self.train_users, self.train_positives, self.train_negatives = \
                self.generate_negatives(pop_alpha=self.bpr_pop_alpha, random_seed=random_seed)

        if only_items:
            feed_dict = {
                self.batch_size_ph: 32,
                self.train_items_pos_ph: np.arange(self.n_items, dtype=np.int32),
                self.train_items_neg_ph: np.ones((self.n_items, self.neg_samples), dtype=np.int32),
                self.train_users_ph: np.ones(self.n_items, dtype=np.int32),
            }
        else:
            feed_dict = {
                self.batch_size_ph: self.batch_size,
                self.train_users_ph: self.train_users,
                self.train_items_pos_ph: self.train_items[self.train_positives],
                self.train_items_neg_ph: self.train_items[self.train_negatives],
            }

        return feed_dict




class NeuralFeatureCombinerProfileBPR_OptimizerMask(NeuralFeatureCombinerProfileBPR):

    def fit(self, topK=200, epochs=30, batch_size=128, l2_reg=1e-4,
            dropout=0.2, rnd_seed=42, activation_function="relu", learning_rate=0.001,
            optimizer=None, bpr_pop_alpha=0., min_item_ratings=1, max_item_ratings=None, checkpoint_path=None,
            log_path=None, evaluator_object=None, stop_on_validation=True, validation_every_n=None,
            lower_validations_allowed=5, validation_metric="RECALL", clear_session_after_train=True, **kwargs):

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

        super(NeuralFeatureCombinerProfileBPR_OptimizerMask, self).fit(
            encoder_layers=[int(value) for key, value in sorted(encoder_layers.items())],
            decoder_layers=[int(value) for key, value in sorted(decoder_layers.items())],
            topK=topK, epochs=epochs, batch_size=batch_size, dropout=dropout, l2_reg=l2_reg,
            rnd_seed=rnd_seed, activation_function=activation_function, bpr_pop_alpha=bpr_pop_alpha,
            learning_rate=learning_rate, optimizer=optimizer, optimizer_args=optimizer_args,
            min_item_ratings=min_item_ratings, max_item_ratings=max_item_ratings, checkpoint_path=checkpoint_path,
            log_path=log_path, evaluator_object=evaluator_object, validation_every_n=validation_every_n,
            lower_validations_allowed=lower_validations_allowed, validation_metric=validation_metric,
            stop_on_validation=stop_on_validation, clear_session_after_train=clear_session_after_train)




class NeuralFeatureCombinerProfileBPRMax_OptimizerMask(NeuralFeatureCombinerProfileBPRMax):

    def fit(self, topK=200, epochs=30, batch_size=128, l2_reg=1e-4, neg_samples=5,
            dropout=0.2, rnd_seed=42, activation_function="relu", learning_rate=0.001,
            optimizer=None, bpr_pop_alpha=0., min_item_ratings=1, max_item_ratings=None, checkpoint_path=None,
            log_path=None, evaluator_object=None, stop_on_validation=True, validation_every_n=None,
            lower_validations_allowed=5, validation_metric="RECALL", clear_session_after_train=True, **kwargs):

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

        super(NeuralFeatureCombinerProfileBPRMax_OptimizerMask, self).fit(
            encoder_layers=[int(value) for key, value in sorted(encoder_layers.items())],
            decoder_layers=[int(value) for key, value in sorted(decoder_layers.items())],
            topK=topK, epochs=epochs, batch_size=batch_size, dropout=dropout, l2_reg=l2_reg, neg_samples=neg_samples,
            rnd_seed=rnd_seed, activation_function=activation_function, bpr_pop_alpha=bpr_pop_alpha,
            learning_rate=learning_rate, optimizer=optimizer, optimizer_args=optimizer_args,
            min_item_ratings=min_item_ratings, max_item_ratings=max_item_ratings, checkpoint_path=checkpoint_path,
            log_path=log_path, evaluator_object=evaluator_object, validation_every_n=validation_every_n,
            lower_validations_allowed=lower_validations_allowed, validation_metric=validation_metric,
            stop_on_validation=stop_on_validation, clear_session_after_train=clear_session_after_train)



