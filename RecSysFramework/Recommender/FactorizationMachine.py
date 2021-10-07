import numpy as np
import scipy.sparse as sps
import subprocess
import multiprocessing
import os, glob
import tempfile
import hashlib

from tqdm import tqdm

from RecSysFramework.Recommender import Recommender
from RecSysFramework.Recommender import ItemSimilarityMatrixRecommender, BaseItemCBFRecommender
from RecSysFramework.Recommender.DataIO import DataIO

from RecSysFramework.Utils import EarlyStoppingModel, check_matrix, urm_to_coordinate_list



class FactorizationMachine(Recommender, EarlyStoppingModel):

    RECOMMENDER_NAME = "FactorizationMachine"
    XLEARN_PATH = "./RecSysFramework/Utils/xlearn/build/"


    def __init__(self, URM_train, ICM=None, UCM=None):

        super(FactorizationMachine, self).__init__(URM_train)

        self.model_file = None

        if UCM is None:
            self.UCM = sps.eye(self.n_users)
        else:
            self.UCM = UCM.copy()
        self.UCM = self.UCM.tocsr().astype(np.float32)

        if ICM is None:
            self.ICM = sps.eye(self.n_items)
        else:
            self.ICM = ICM.copy()
        self.ICM = self.ICM.tocsr().astype(np.float32)

        self._create_output_strings()


    def _create_output_strings(self, fields=None):

        value_pattern = u"%d:%.8g"

        if fields is not None:
            is_ffm_format = True
            value_pattern = u"%d:" + value_pattern
        else:
            is_ffm_format = False

        self.UCM_strings = [None] * self.n_users
        self.ICM_strings = [None] * self.n_items

        for i in range(self.n_users):

            span = slice(self.UCM.indptr[i], self.UCM.indptr[i + 1])
            x_indices = self.UCM.indices[span]
            row = zip(fields[x_indices], x_indices, self.UCM.data[span]) if is_ffm_format \
                else zip(x_indices, self.UCM.data[span])

            if is_ffm_format:
                self.UCM_strings[i] = " ".join(value_pattern % (f, j, x) for f, j, x in row)
            else:
                self.UCM_strings[i] = " ".join(value_pattern % (j, x) for j, x in row)

        for i in range(self.n_items):

            span = slice(self.ICM.indptr[i], self.ICM.indptr[i + 1])
            x_indices = self.ICM.indices[span]
            row = zip(fields[x_indices], x_indices, self.ICM.data[span]) if is_ffm_format \
                else zip(x_indices, self.ICM.data[span])

            if is_ffm_format:
                self.ICM_strings[i] = " ".join(value_pattern % (f, j, x) for f, j, x in row)
            else:
                self.ICM_strings[i] = " ".join(value_pattern % (j, x) for j, x in row)



    def write_data_to_xlearn_format(self, users, items, y, filepath):
        """ Write data to xlearn format (libsvm or libffm). Modified from
        https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/datasets/svmlight_format.py

        :param X: array-like
                  Feature matrix in numpy or sparse format
        :param y: array-like
                  Label in numpy or sparse format
        :param filepath: file location for writing data to
        :param fields: An array specifying fields in each columns of X. It should have same length
            as the number of columns in X. When set to None, convert data to libsvm format else
            libffm format.
        """

        y_is_sp = int(hasattr(y, "tocsr"))

        with open(filepath, "wb") as f_handle:

            if y.dtype.kind == 'i':
                label_pattern = u"%d"
            else:
                label_pattern = u"%.8g"

            line_pattern = u"%s %s\n"

            for i in tqdm(range(len(users)), desc="Writing data file"):

                s = self.UCM_strings[users[i]] + " " + self.ICM_strings[items[i]]

                if y_is_sp:
                    labels_str = label_pattern % y.data[i]
                else:
                    labels_str = label_pattern % y[i]

                f_handle.write((line_pattern % (labels_str, s)).encode('ascii'))


    def get_model_filename(self):
        return self.model_file if isinstance(self.model_file, str) else self.model_file.name


    def _compute_item_score(self, user_id_array, items_to_compute=None):

        if not isinstance(user_id_array, np.ndarray):
            user_id_array = np.array(user_id_array)

        if items_to_compute is None:
            items_to_compute = np.arange(self.n_items, dtype=np.int32)

        predictions = np.zeros((user_id_array.size, self.n_items), dtype=np.float32)

        users = np.repeat(user_id_array, len(items_to_compute))
        items = np.tile(items_to_compute, len(user_id_array))

        temp_out_file = tempfile.NamedTemporaryFile(delete=False, dir=self.tmp_files_dir)

        hash = hashlib.sha256(bytes("".join(map(str, np.sort(user_id_array).tolist())), encoding='utf8'))
        temp_test_file = self.temp_test_basefile + hash.hexdigest()

        if not (os.path.exists(temp_test_file + ".bin") or os.path.exists(temp_test_file)):
            self.write_data_to_xlearn_format(users, items, np.zeros(len(users), dtype=np.float32), temp_test_file)

        args = [
            "{}xlearn_predict".format(self.XLEARN_PATH),
            temp_test_file,
            self.get_model_filename(),
            "-o", "{}".format(temp_out_file.name),
            "-nthread", "{}".format(multiprocessing.cpu_count()),
        ]

        subprocess.run(args)

        predictions[:, items_to_compute] = np.reshape(np.loadtxt(temp_out_file.name), (-1, len(items_to_compute)))

        self._remove_temp_file(temp_out_file)

        return predictions


    def fit(self, rank=10, epochs=30, reg=1e-4, learning_rate=0.001, add_zeros_quota=2.0, normalize_instance=True,
            problem_type="regression", n_threads=None, optimizer=None, model_file=None,
            tmp_files_dir=None, **earlystopping_kwargs):

        if optimizer is None:
            optimizer = "adagrad"

        if n_threads is None:
            n_threads = multiprocessing.cpu_count()

        self.tmp_files_dir = tmp_files_dir
        if self.tmp_files_dir is not None:
            os.makedirs(self.tmp_files_dir, exist_ok=True)

        if problem_type == "classification":
            self.problem_type = 1
        else:
            self.problem_type = 4

        self.add_zeros_quota = add_zeros_quota
        self.rank = rank
        self.reg = reg
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.n_threads = n_threads
        self.normalize_instance = normalize_instance

        self.temp_test_basefile = tempfile.NamedTemporaryFile(delete=False, dir=self.tmp_files_dir).name
        self.temp_train_file = tempfile.NamedTemporaryFile(delete=False, dir=self.tmp_files_dir)
        if model_file is None:
            self.model_file = tempfile.NamedTemporaryFile(delete=False, dir=self.tmp_files_dir)
        else:
            self.model_file = model_file

        self._train_with_early_stopping(epochs, algorithm_name=self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)

        self._remove_temp_file(self.temp_train_file)


    def _run_multiple_epochs(self, num_epoch, epochs_to_run):

        if num_epoch % 5 == 0:
            self._remove_temp_file(self.temp_train_file)
            users, items, y = urm_to_coordinate_list(self.URM_train, self.add_zeros_quota, random_seed=num_epoch)
            self.write_data_to_xlearn_format(users, items, y, self.temp_train_file.name)

        args = [
            "{}xlearn_train".format(self.XLEARN_PATH),
            self.temp_train_file.name,
            "-s", "{}".format(self.problem_type),
            "-m", "{}".format(self.get_model_filename()),
            "-p", "{}".format(self.optimizer),
            "-r", "{}".format(self.learning_rate),
            "-b", "{}".format(self.reg),
            "-k", "{}".format(self.rank),
            "-e", "{}".format(epochs_to_run),
            "--quiet",
            "-nthread", "{}".format(self.n_threads),
        ]

        if not self.normalize_instance:
            args.append("--no-norm")

        if num_epoch > 0:
            args.extend(["-pre", self.get_model_filename()])

        subprocess.run(args)


    def _run_epoch(self, num_epoch):
        self._run_multiple_epochs(num_epoch, 1)


    def _prepare_model_for_validation(self):
        #Nothing to do
        return


    def _update_best_model(self):
        #Nothing to do
        return


    def _remove_temp_file(self, temp_file, is_prefix=False):
        # The temp_file might be converted to binary file during training/inference.
        # remove both original temp_file and derived binary file if exist

        if isinstance(temp_file, str):
            temp_filename = temp_file
        else:
            temp_filename = temp_file.name
            temp_file.close()

        if is_prefix:
            for filename in glob.glob(temp_filename + "*"):
                os.remove(filename)
        else:
            temp_bin_file = temp_filename + '.bin'
            if os.path.exists(temp_bin_file):
                os.remove(temp_bin_file)
            if os.path.exists(temp_filename):
                os.remove(temp_filename)


    def save_model(self, folder_path, file_name=None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        new_filename = folder_path + self.RECOMMENDER_NAME + "_xlearn_model"
        os.replace(self.get_model_filename(), new_filename)
        data_dict_to_save = {"model_file": new_filename}

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save=data_dict_to_save)

        self._print("Saving complete")


    def __del__(self):
        del self.ICM
        del self.UCM
        del self.URM_train
        self._remove_temp_file(self.temp_test_basefile, is_prefix=True)
        if self.model_file is not None:
            if not isinstance(self.model_file, str):
                self._remove_temp_file(self.model_file)



class FactorizationMachineProfile(FactorizationMachine):

    RECOMMENDER_NAME = "FactorizationMachineProfile"

    def __init__(self, URM_train, ICM=None):
        super(FactorizationMachineProfile, self).__init__(URM_train, ICM=ICM, UCM=URM_train)



class FactorizationMachineSimilarity(FactorizationMachine, ItemSimilarityMatrixRecommender, BaseItemCBFRecommender):

    RECOMMENDER_NAME = "FactorizationMachineSimilarity"

    def __init__(self, URM_train, ICM, S_matrix=None):
        super(FactorizationMachineSimilarity, self).__init__(URM_train, ICM=ICM, UCM=None)
        self.S_matrix = check_matrix(S_matrix, 'csr')



    def __del__(self):
        del self.S_matrix
        super(FactorizationMachineSimilarity, self).__del__()
