import numpy as np
import os, pickle, zipfile
import glob
import tensorflow as tf
import argparse

import scipy.sparse as sps
import similaripy as sim
import pandas as pd

from sklearn.preprocessing import normalize

from deepexplain.tensorflow import DeepExplain

from RecSysFramework.Recommender.KNN import ItemKNNCF, ItemKNNCBF, CFW_D, EASE_R
from RecSysFramework.Recommender.SLIM.ElasticNet import SLIM
from RecSysFramework.Recommender.GraphBased import RP3beta, HP3
from RecSysFramework.Recommender.MatrixFactorization import LCE, BPRMF_AFM

from RecSysFramework.Recommender.DeepLearning import WideAndDeep
from RecSysFramework.Recommender.DeepLearning import NeuralFeatureCombinerFW
from RecSysFramework.Recommender.DeepLearning import NeuralFeatureCombiner_OptimizerMask as NeuralFeatureCombiner
from RecSysFramework.Recommender.DeepLearning import NeuralFeatureCombinerProfile_OptimizerMask as NeuralFeatureCombinerProfile
from RecSysFramework.Recommender.DeepLearning import NeuralFeatureCombinerProfileBPR_OptimizerMask as NeuralFeatureCombinerProfileBPR

from evaluate_on_test import find_best_configuration, train_best_config, algorithm_name_to_class

from RecSysFramework.DataManager.Splitter import ColdItemsHoldout, Holdout
from RecSysFramework.DataManager import Dataset

from RecSysFramework.Recommender.DataIO import DataIO

from RecSysFramework.DataManager.Reader import YahooMoviesReader, YahooMoviesReducedReader
from RecSysFramework.Evaluation import EvaluatorHoldout
from RecSysFramework.ExperimentalConfig import EXPERIMENTAL_CONFIG


parser = argparse.ArgumentParser(description='Running embeddings qualitative analysis of NeuralFeatureCombiner')
parser.add_argument('--nosynopsis', dest="nosynopsis", const=True, default=False, nargs="?",
                    help='If specified, uses noisy version of YahooMovies without synopsis')
parser.add_argument('--noisy', dest="noisy", const=True, default=False, nargs="?",
                    help='If specified, runs the analysis removing progressively random features simulating noise')
parser.add_argument('--genrefilter', dest="genrefilter", default=[2], nargs=1, type=int, choices=[0, 1, 2],
                    help='Determines the strength of the filter to apply to genres')

arguments = parser.parse_args()

_NOISY = arguments.noisy
_NO_SYNOPSIS = arguments.nosynopsis
_GENRE_FILTER = arguments.genrefilter[0]


dataset_config = EXPERIMENTAL_CONFIG['datasets'][0]
ym_reader = YahooMoviesReader()
ymred_reader = YahooMoviesReducedReader()
if _NO_SYNOPSIS:
    datareader = ymred_reader
    _FW_FILENAME = "fwdata-nosyn.csv"
else:
    datareader = ym_reader
    _FW_FILENAME = "fwdata-syn.csv"

postprocessings = dataset_config['postprocessings']
splitter = EXPERIMENTAL_CONFIG['cold_split']

warm_split = EXPERIMENTAL_CONFIG['warm_split']
collaborative_splitter = EXPERIMENTAL_CONFIG['warm_split']

basepath = splitter.get_complete_default_save_folder_path(ym_reader, postprocessings=postprocessings)
cold_basepath = basepath + splitter.get_name() + os.sep
collaborative_basepath = cold_basepath + collaborative_splitter.get_name() + os.sep

train, test = splitter.load_split(datareader, postprocessings=postprocessings)

URM_train = train.get_URM() + test.get_URM()
# The ICM is always the same in all the splits
ICM_object = train.get_ICM().tocsr()
#feature_mapper = dict((str(k), v) for k, v in train.get_ICM_mapper()[1].items())
feature_mapper = train.get_ICM_mapper()[1]
inv_feature_mapper = {v: str(k) for k, v in feature_mapper.items()}
feature_mapper = {str(k): str(v) for k, v in feature_mapper.items()}

item_mapper = dict((str(k), v) for k, v in train.get_ICM_mapper()[0].items())
inv_item_mapper = {v: str(k) for k, v in item_mapper.items()}

zipFile_path = datareader.DATASET_SPLIT_ROOT_FOLDER + datareader.DATASET_SUBFOLDER 

try:

    dataFile = zipfile.ZipFile(ym_reader.DATASET_OFFLINE_ROOT_FOLDER + \
            ym_reader.DATASET_SUBFOLDER + "yahoo-movies-dataset.zip")

except (FileNotFoundError, zipfile.BadZipFile):

    print("YahooMovies: Unable to find data zip file.")
    print("YahooMovies: Automatic download not available, " \
          "please ensure the ZIP data file is in folder {}.".format(zipFile_path))
    print("YahooMovies: Data can be downloaded here: {}".format(datareader.DATASET_URL))

    # If directory does not exist, create
    if not os.path.exists(zipFile_path):
        os.makedirs(zipFile_path)

    raise FileNotFoundError("Automatic download not available.")

ICM_path = dataFile.extract("dataset/ydata-ymovies-movie-content-descr.txt", path=zipFile_path + "decompressed/")

fileHandle = open(ICM_path, "r", encoding="latin1")
fileHandle.readline()

raw_item_features = {}
for line in fileHandle:
    if len(line) > 1:
        line = line.split('\t')
        line[-1] = line[-1].replace("\n", "")
        movie_id = str(line[0])
        raw_item_features[movie_id] = line[1:]
        raw_item_features[movie_id][9] = raw_item_features[movie_id][9].lower().strip().split("|")
        del raw_item_features[movie_id][1]

_GENRES = [
  ('suspense/horror', 3),
  ('thriller', 3),
  ('crime/gangster', 2),
  ('western', 1),
  ('drama', 3),
  ('romance', 3),
  ('comedy', 3),
  ('action/adventure', 3),
  ('science fiction/fantasy', 2),
  ('kids/family', 2),
  ('animation', 1),
  ('musical/performing arts', 1),
  ('art/foreign', 1),
  ('documentary', 1)
]

with open(zipFile_path + "directors_name_to_id.pkl", "rb") as file:
    directors_name_to_id = pickle.load(file)
    for k in directors_name_to_id.keys():
        directors_name_to_id[k] = str(directors_name_to_id[k])
with open(zipFile_path + "actors_name_to_id.pkl", "rb") as file:
    actors_name_to_id = pickle.load(file)
    for k in actors_name_to_id.keys():
        actors_name_to_id[k] = str(actors_name_to_id[k])

fm_keys = list(map(str, feature_mapper.keys()))

JBMOVIES_YIDS = ['1800249529', '1800066832', '1800127497', '1800059368', '1800096934', '1800071536', '1800092955',
                 '1800102255', '1800120663', '1800137846', '1800132578', '1800088536', '1800089890', '1800381592']
IJMOVIES_YIDS = ['1800080788', '1800080795', '1800024141']
JBMOVIES_IDS = np.array([item_mapper[yid] for yid in JBMOVIES_YIDS])
IJMOVIES_IDS = np.array([item_mapper[yid] for yid in IJMOVIES_YIDS])

_CREW_MAPPING = {}
for id in range(URM_train.shape[1]):
    features = raw_item_features[inv_item_mapper[id]]
    keys = features[13].split("|")
    values = features[12].split("|")
    for k,v in zip(keys, values):
        _CREW_MAPPING[k] = v.lower().strip()


inv_actors_name_to_id = dict((v, k) for k, v in actors_name_to_id.items())
inv_directors_name_to_id = dict((v, k) for k, v in directors_name_to_id.items())
def feature_name(f):
    n = inv_feature_mapper[f]
    if "18000" in n:
        key = n.split("_")[0]
        if key in inv_actors_name_to_id.keys():
            n = inv_actors_name_to_id[key]
        elif key in inv_directors_name_to_id.keys():
            n = inv_directors_name_to_id[key]
        elif key in _CREW_MAPPING.keys():
            n = _CREW_MAPPING[key]
    return n


def cosine(a, b):
    if sps.issparse(a):
        numerator = a.dot(b.T).toarray()
        normf = sps.linalg.norm
    else:
        numerator = np.matmul(a, b.T)
        normf = np.linalg.norm
    denominator = np.outer(normf(a, axis=-1).flatten(), normf(b, axis=-1).flatten())
    return np.divide(numerator, denominator)


def show_item_neighbors(items=None, neighbors_ids=None, toshow=5):

    if items is None:
        items = ["1800080795", "1800381592"]
        neighbors_ids = [IJMOVIES_IDS, JBMOVIES_IDS]

    for movie_id, movies_ids in zip(items, neighbors_ids):

        if isinstance(movie_id, str):
            movie_id = item_mapper[movie_id]

        movie_features = ICM_object[movie_id, :]

        n_items = URM_train.shape[1]
        test_movies = np.union1d(np.random.choice(n_items, int(0.2 * n_items), replace=False), movies_ids)
        test_movies = test_movies[test_movies != movie_id]

        URM_train_filtered = URM_train.copy()
        ICM_train_filtered = ICM_object.copy()

        URM_train_filtered[:, test_movies] = 0
        URM_train_filtered.eliminate_zeros()

        ICM_train_filtered[test_movies, :] = 0
        ICM_train_filtered.eliminate_zeros()

        dataset = Dataset("Yahoo-no-" + str(movie_id),
            URM_dict={"URM_all": URM_train_filtered}, URM_mappers_dict=train.get_URM_mappers_dict(),
            ICM_dict={"ICM_all": ICM_train_filtered}, ICM_mappers_dict=train.get_ICM_mappers_dict())

        for recommender_class in [ItemKNNCBF, LCE, BPRMF_AFM, CFW_D, NeuralFeatureCombiner]:

            W_train = None
            icm_name = "ICM_all"
            additional_parameters = {}
            recommender_name = recommender_class.RECOMMENDER_NAME

            if recommender_class in [CFW_D, NeuralFeatureCombiner]:

                def fit_collaborative_algorithm(_collaborative_algorithm, _urm):
                    algo_basepath = cold_basepath + _collaborative_algorithm.RECOMMENDER_NAME + os.sep
                    dataIO = DataIO(folder_path=algo_basepath)
                    data_dict = dataIO.load_data(file_name=_collaborative_algorithm.RECOMMENDER_NAME + "_metadata")
                    recommender = _collaborative_algorithm(_urm)
                    recommender.fit(**data_dict["hyperparameters_best"])
                    W_train = recommender.get_W_sparse()
                    del recommender
                    return W_train

                output_folder_path = find_best_configuration(basepath, recommender_name,
                                                         add_underscore="NeuralFeatureCombiner" in recommender_name)
                collaborative_algorithm = algorithm_name_to_class(output_folder_path.split(os.sep)[-3])
                W_train = fit_collaborative_algorithm(collaborative_algorithm, URM_train_filtered)
            else:
                output_folder_path = find_best_configuration(basepath, recommender_name,
                                                         add_underscore="NeuralFeatureCombiner" in recommender_name)

            if recommender_class in [NeuralFeatureCombiner, NeuralFeatureCombinerProfile, NeuralFeatureCombinerProfileBPR]:
                graph = tf.Graph()
                config = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                                  intra_op_parallelism_threads=0,
                                                  inter_op_parallelism_threads=0)
                config.gpu_options.allow_growth = True

                session = tf.compat.v1.Session(config=config, graph=graph)

                #additional_parameters['checkpoint_path'] = zipFile_path + "tf_train_cold_{}_{}".format(movie_id, recommender_name)
                #additional_parameters['log_path'] = additional_parameters['checkpoint_path'] + os.sep + "logs" + os.sep
                additional_parameters['clear_session_after_train'] = False
                additional_parameters['graph'] = graph
                additional_parameters['session'] = session

            recommender, _ = train_best_config(recommender_class, dataset, output_folder_path,
                            icm_name=icm_name, W_train=W_train, additional_parameters=additional_parameters)

            embedded_movie = recommender.get_embedding(movie_features)
            embedded_icm = recommender.get_embedding(ICM_object[test_movies, :])

            sim_mat = cosine(embedded_movie, embedded_icm)
            most_similar = np.argsort(-sim_mat)[:, :toshow]
            most_similar = test_movies[most_similar].flatten()

            print("{}:".format(recommender_name))
            for i in most_similar.tolist():
                print(raw_item_features[inv_item_mapper[i]][0])



if __name__ == "__main__":
    show_item_neighbors()


