import numpy as np
import scipy.sparse as sps
import tensorflow as tf
import os
import logging
import math

from functools import partial

from RecSysFramework.Recommender import Recommender
from RecSysFramework.Utils import EarlyStoppingModel, urm_to_coordinate_list

logging.getLogger('tensorflow').disabled = True


class ProgressbarRunHook(tf.estimator.SessionRunHook):

    def __init__(self, steps, width=50, verbose=1):
        super(ProgressbarRunHook, self).__init__()
        self.pbar = None
        self.steps = steps
        self.width = width
        self.verbose = verbose
        self.pbar = tf.keras.utils.Progbar(self.steps, width=self.width, verbose=self.verbose)

    def begin(self):
        """Called once before using the session.
        When called, the default graph is the one that will be launched in the
        session.  The hook can modify the graph by adding new operations to it.
        After the `begin()` call the graph will be finalized and the other callbacks
        can not modify the graph anymore. Second call of `begin()` on the same
        graph, should not change the graph.
        """

    def after_run(self, run_context, run_values):
        """Called after each call to run().
        The `run_values` argument contains results of requested ops/tensors by
        `before_run()`.
        The `run_context` argument is the same one send to `before_run` call.
        `run_context.request_stop()` can be called to stop the iteration.
        If `session.run()` raises any exceptions then `after_run()` is not called.
        Args:
          run_context: A `SessionRunContext` object.
          run_values: A SessionRunValues object.
        """
        self.pbar.add(1)

    def end(self, session):
        """Called at the end of session.
        The `session` argument can be used in case the hook wants to run final ops,
        such as saving a last checkpoint.
        If `session.run()` raises exception other than OutOfRangeError or
        StopIteration then `end()` is not called.
        Note the difference between `end()` and `after_run()` behavior when
        `session.run()` raises OutOfRangeError or StopIteration. In that case
        `end()` is called but `after_run()` is not called.
        Args:
          session: A TensorFlow Session that will be soon closed.
        """



class WideAndDeep(Recommender, EarlyStoppingModel):

    RECOMMENDER_NAME = "WideAndDeep"


    def __init__(self, URM_train, ICM=None, UCM=None):

        super(WideAndDeep, self).__init__(URM_train)
        self.item_popularities = np.ediff1d(self.URM_train.tocsc().indptr)

        if UCM is None:
            self.UCM = sps.eye(self.n_users).tocsr()
            self.UCM_wide = False
        else:
            self.UCM = sps.csr_matrix(UCM)
            self.UCM_wide = True

        self.UCM_feature_sizes = np.ediff1d(self.UCM.tocsr().indptr)

        if ICM is None:
            self.ICM = sps.eye(self.n_items).tocsr()
            self.ICM_wide = False
        else:
            self.ICM = sps.csr_matrix(ICM)
            self.ICM_wide = True


    def _compute_item_score(self, user_id_array, items_to_compute=None):

        if not isinstance(user_id_array, np.ndarray):
            user_id_array = np.array(user_id_array)

        batch_size = 4
        if items_to_compute is None:
            items_to_compute = np.arange(self.n_items, dtype=np.int32)

        actual_padded_item_features = self.padded_item_features[items_to_compute, :]
        actual_padded_user_features = np.atleast_2d(self.padded_user_features[user_id_array, :])

        def input_fn_predict():

            user_dataset = tf.data.Dataset.from_tensor_slices(actual_padded_user_features)
            user_dataset = user_dataset.prefetch(buffer_size=4)

            items_dataset = tf.data.Dataset.from_tensors(actual_padded_item_features).cache()
            items_dataset = items_dataset.repeat(len(user_id_array))

            def tf_wrapper(u, f):
                return {'user_features': tf.repeat(u, len(items_to_compute), axis=0),
                        'item_features': tf.reshape(f, [-1, actual_padded_item_features.shape[1]])}

            dataset = tf.data.Dataset.zip((user_dataset, items_dataset))
            dataset = dataset.batch(batch_size)
            dataset = dataset.map(tf_wrapper, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.prefetch(buffer_size=2)

            return dataset

        predictions = np.ones((user_id_array.size, self.n_items), dtype=np.float32) * -np.inf
        steps = math.ceil(user_id_array.size / batch_size)
        hook = ProgressbarRunHook(steps)

        ub = 0
        for u, p in enumerate(self.estimator.predict(input_fn=input_fn_predict, hooks=[hook], yield_single_examples=False)):
            bsize = int(p['predictions'].size / len(items_to_compute))
            predictions[ub:ub+bsize, :][:, items_to_compute] = p['predictions'].reshape((bsize, len(items_to_compute)))
            ub += bsize

        return predictions


    def fit(self, layers=None, epochs=50, user_embedding_size=64, item_embedding_size=64,
            batch_size=32, dropout=0.2, hash_bucket_size=1000, add_zeros_quota=2., learning_rate=0.001,
            model_dir=None, optimizer=None, optimizer_args=None, **earlystopping_kwargs):

        if layers is None:
            layers = [1024, 512, 256]
        self.layers = layers

        if model_dir is not None:
            os.makedirs(model_dir, exist_ok=True)

        self.batch_size = batch_size
        self.add_zeros_quota = add_zeros_quota

        if optimizer is None:
            optimizer = "adagrad"

        if optimizer_args is None:
            optimizer_args = {}

        if optimizer == "adam":
            # Tensorflow is bugged... issue #33358
            dnn_optimizer = partial(tf.keras.optimizers.Adam, learning_rate=learning_rate, **optimizer_args)
            linear_optimizer = partial(tf.keras.optimizers.Adam, learning_rate=learning_rate, **optimizer_args)
        elif optimizer == "adagrad":
            dnn_optimizer = partial(tf.keras.optimizers.Adagrad, learning_rate=learning_rate, **optimizer_args)
            linear_optimizer = partial(tf.keras.optimizers.Adagrad, learning_rate=learning_rate, **optimizer_args)
        elif optimizer == "adadelta":
            dnn_optimizer = partial(tf.keras.optimizers.Adadelta, learning_rate=learning_rate, **optimizer_args)
            linear_optimizer = partial(tf.keras.optimizers.Adadelta, learning_rate=learning_rate, **optimizer_args)
        elif optimizer == "rmsprop":
            dnn_optimizer = partial(tf.keras.optimizers.RMSprop, learning_rate=learning_rate, **optimizer_args)
            linear_optimizer = partial(tf.keras.optimizers.RMSprop, learning_rate=learning_rate, **optimizer_args)
        elif optimizer == "sgd":
            dnn_optimizer = partial(tf.keras.optimizers.SGD, learning_rate=learning_rate, **optimizer_args)
            linear_optimizer = partial(tf.keras.optimizers.SGD, learning_rate=learning_rate, **optimizer_args)
        else:
            raise ValueError("{}: Unknown value of optimizer".format(self.RECOMMENDER_NAME))

        self.train_items = np.arange(self.n_items, dtype=np.int32)[np.ediff1d(self.URM_train.tocsc().indptr) > 0]
        ifeat_tokeep = np.ediff1d(self.ICM[self.train_items, :].tocsc().indptr) > 0
        ICM = self.ICM[:, np.arange(self.ICM.shape[1])[ifeat_tokeep]]

        self.train_users = np.arange(self.n_users, dtype=np.int32)[np.ediff1d(self.URM_train.tocsr().indptr) > 0]
        ufeat_tokeep = np.ediff1d(self.UCM[self.train_users, :].tocsc().indptr) > 0
        UCM = self.UCM[:, np.arange(self.UCM.shape[1])[ufeat_tokeep]]

        self._pad_user_features(UCM)
        self._pad_item_features(ICM)

        user_features = tf.feature_column.categorical_column_with_vocabulary_list("user_features",
                                                                                  np.arange(UCM.shape[1]),
                                                                                  dtype=tf.int32)

        item_features = tf.feature_column.categorical_column_with_vocabulary_list("item_features",
                                                                                  np.arange(ICM.shape[1]),
                                                                                  dtype=tf.int32)

        session_config = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                        intra_op_parallelism_threads=0,
                                        inter_op_parallelism_threads=0)
        session_config.gpu_options.allow_growth = True
        config = tf.estimator.RunConfig(session_config=session_config)

        features_to_cross = []
        linear_features_shape = 0

        if self.ICM_wide:
            features_to_cross.append(item_features)
            linear_features_shape += ICM.shape[1]

        if self.UCM_wide:
            features_to_cross.append(user_features)
            linear_features_shape += UCM.shape[1]

        dnn_features = [
            tf.feature_column.embedding_column(user_features, user_embedding_size),
            tf.feature_column.embedding_column(item_features, item_embedding_size),
        ]

        if len(features_to_cross) > 0:

            linear_features = features_to_cross
            if len(features_to_cross) > 1:
                linear_features.append(tf.feature_column.crossed_column(features_to_cross, hash_bucket_size))

            self.estimator = tf.estimator.DNNLinearCombinedRegressor(
                # wide settings
                linear_feature_columns=linear_features,
                linear_optimizer=linear_optimizer,
                # deep settings
                dnn_feature_columns=dnn_features,
                dnn_hidden_units=self.layers,
                config=config,
                model_dir=model_dir,
                dnn_dropout=dropout,
                dnn_optimizer=dnn_optimizer, loss_reduction=tf.keras.losses.Reduction.SUM)
        else:
            self.estimator = tf.estimator.DNNRegressor(
                # deep settings
                feature_columns=dnn_features,
                hidden_units=self.layers,
                dropout=dropout,
                config=config,
                model_dir=model_dir,
                optimizer=dnn_optimizer, loss_reduction=tf.keras.losses.Reduction.SUM)

        def input_fn_train(dataset_row, dataset_col, dataset_data):

            def gen_sample(x):
                user, item = dataset_row[x], dataset_col[x]
                return self.padded_user_features[user, :], \
                       self.padded_item_features[item, :], \
                       np.atleast_2d(dataset_data[x]).T

            def tf_wrapper(x):
                features = tf.compat.v1.py_func(func=gen_sample,
                                      inp=[x],
                                      stateful=False,
                                      Tout=[tf.int32, tf.int32, tf.float32])
                return {'user_features': features[0],
                        'item_features': features[1]}, \
                       features[2]

            dataset = tf.data.Dataset.range(dataset_data.size)

            dataset = dataset.repeat().shuffle(buffer_size=5000)
            dataset = dataset.batch(batch_size)
            dataset = dataset.map(tf_wrapper, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

            return dataset

        # This number of steps is not the correct one, it is higher for sure (it does not count overlaps)
        # It is not a real problem...

        self.input_fn_train = input_fn_train

        self._train_with_early_stopping(epochs, algorithm_name=self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)

        print("{}: Training concluded".format(self.RECOMMENDER_NAME))


    def _run_multiple_epochs(self, num_epoch, epochs_to_run):

        if num_epoch % 3 == 0:
            self.dataset_row, self.dataset_col, self.dataset_data = \
                urm_to_coordinate_list(self.URM_train, self.add_zeros_quota, random_seed=num_epoch)

        final_datasize = len(self.dataset_row)
        steps = int(epochs_to_run * final_datasize / self.batch_size) + 1
        hook = ProgressbarRunHook(steps)
        self.estimator.train(input_fn=lambda: self.input_fn_train(self.dataset_row, self.dataset_col, self.dataset_data),
                             steps=steps, hooks=[hook])


    def _run_epoch(self, num_epoch):
        self._run_multiple_epochs(num_epoch, 1)


    def _prepare_model_for_validation(self):
        #Nothing to do
        return


    def _update_best_model(self):
        #Nothing to do
        return


    def _pad_item_features(self, ICM):
        max_icm_features = np.ediff1d(ICM.tocsr().indptr).max()
        self.padded_item_features = []
        for item in range(self.n_items):
            ifeat = ICM.indices[ICM.indptr[item]:ICM.indptr[item+1]]
            self.padded_item_features.append(np.pad(ifeat, (0, max_icm_features - ifeat.size),
                                                    "constant", constant_values=-1))
        self.padded_item_features = np.vstack(self.padded_item_features)


    def _pad_user_features(self, UCM):
        max_ucm_features = np.ediff1d(UCM.tocsr().indptr).max()
        self.padded_user_features = []
        for user in range(self.n_users):
            ufeat = UCM.indices[UCM.indptr[user]:UCM.indptr[user + 1]]
            self.padded_user_features.append(np.pad(ufeat, (0, max_ucm_features - ufeat.size),
                                                    "constant", constant_values=-1))
        self.padded_user_features = np.vstack(self.padded_user_features)



class WideAndDeep_OptimizerMask(WideAndDeep):

    def fit(self, layers=None, epochs=50, user_embedding_size=64, item_embedding_size=64, batch_size=32, dropout=0.2,
            add_zeros_quota=1.0, hash_bucket_size=1000, learning_rate=0.001, optimizer=None, evaluator_object=None,
            stop_on_validation=True, validation_every_n=None, lower_validations_allowed=5,
            validation_metric="RECALL", **kwargs):

        layers = {}
        for key, value in kwargs.items():
            if "layers_" in key:
                layers[key] = value

        optimizer_args = {}
        for key, value in kwargs.items():
            if "optimizer_args_" in key:
                optimizer_args[key.split("_")[-1]] = value

        super(WideAndDeep_OptimizerMask, self).fit(
            layers=[int(value) for key, value in sorted(layers.items())],
            epochs=epochs, batch_size=batch_size, dropout=dropout,
            user_embedding_size=user_embedding_size, item_embedding_size=item_embedding_size,
            hash_bucket_size=hash_bucket_size, add_zeros_quota=add_zeros_quota, evaluator_object=evaluator_object,
            validation_every_n=validation_every_n, lower_validations_allowed=lower_validations_allowed,
            validation_metric=validation_metric, stop_on_validation=stop_on_validation,
            learning_rate=learning_rate, optimizer=optimizer, optimizer_args=optimizer_args)



class WideAndDeepProfile(WideAndDeep):

    RECOMMENDER_NAME = "WideAndDeepProfile"

    def __init__(self, URM_train, ICM=None):
        super(WideAndDeepProfile, self).__init__(URM_train, ICM=ICM, UCM=URM_train)
        self.force_max_profile_length = np.sort(np.ediff1d(self.UCM.indptr))[int(0.9 * self.n_users)]
        #self.UCM_wide = False


    def _pad_user_features(self, UCM):
        if self.force_max_profile_length is not None:
            max_ucm_features = min(np.ediff1d(UCM.tocsr().indptr).max(), self.force_max_profile_length)
        else:
            max_ucm_features = np.ediff1d(UCM.tocsr().indptr).max()

        self.padded_user_features = []
        for user in range(self.n_users):
            ufeat = UCM.indices[UCM.indptr[user]:UCM.indptr[user + 1]]
            if self.force_max_profile_length is not None and len(ufeat) > self.force_max_profile_length:
                ufeat = ufeat[np.argsort(-self.item_popularities[ufeat])[:self.force_max_profile_length]]
            self.padded_user_features.append(np.pad(ufeat, (0, max_ucm_features - ufeat.size),
                                                    "constant", constant_values=-1))
        self.padded_user_features = np.vstack(self.padded_user_features)



class WideAndDeepProfile_OptimizerMask(WideAndDeepProfile):

    def fit(self, layers=None, epochs=50, user_embedding_size=64, item_embedding_size=64, batch_size=32, dropout=0.2,
            hash_bucket_size=1000, learning_rate=0.001, optimizer=None, evaluator_object=None, stop_on_validation=True,
            validation_every_n=None, lower_validations_allowed=5, validation_metric="RECALL", **kwargs):

        layers = {}
        for key, value in kwargs.items():
            if "layers_" in key:
                layers[key] = value

        optimizer_args = {}
        for key, value in kwargs.items():
            if "optimizer_args_" in key:
                optimizer_args[key.split("_")[-1]] = value

        super(WideAndDeepProfile_OptimizerMask, self).fit(
            layers=[int(value) for key, value in sorted(layers.items())],
            epochs=epochs, batch_size=batch_size, dropout=dropout,
            user_embedding_size=user_embedding_size, item_embedding_size=item_embedding_size,
            hash_bucket_size=hash_bucket_size, evaluator_object=evaluator_object, validation_every_n=validation_every_n,
            lower_validations_allowed=lower_validations_allowed, validation_metric=validation_metric,
            stop_on_validation=stop_on_validation, learning_rate=learning_rate, optimizer=optimizer,
            optimizer_args=optimizer_args)
