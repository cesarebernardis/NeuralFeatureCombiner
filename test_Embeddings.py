import numpy as np
import os, pickle, zipfile
import glob
import tensorflow as tf
import argparse

import scipy.sparse as sps
import similaripy as sim
import seaborn as sns
import matplotlib.pyplot as plt
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

from RecSysFramework.Recommender.DataIO import DataIO

from RecSysFramework.DataManager.Reader import YahooMoviesReader, YahooMoviesReducedReader
from RecSysFramework.Evaluation import EvaluatorHoldout
from RecSysFramework.ExperimentalConfig import EXPERIMENTAL_CONFIG

sns.set(font_scale=1.2)

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
if _NO_SYNOPSIS:
    datareader = YahooMoviesReducedReader()
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
SWMOVIES_YIDS = ['1800121659', '1800421139', '1800379216', '1800061638', '1800111258']
STMOVIES_YIDS = ['1800024802', '1800019361', '1800024819', '1800024816', '1800024806', '1800121143', '1800024823', '1807859445', '1800024813', '1800024791']
NMMOVIES_YIDS = ['1802816835', '1800222436', '1800101006', '1800100997', '1800101018', '1800101011', '1800165355']
HWMOVIES_YIDS = ['1800318700', '1807530862', '1800254903', '1800074095', '1800074071', '1800074089', '1800074038', '1800074058']
JBMOVIES_IDS = np.array([item_mapper[yid] for yid in JBMOVIES_YIDS])
SWMOVIES_IDS = np.array([item_mapper[yid] for yid in SWMOVIES_YIDS])
STMOVIES_IDS = np.array([item_mapper[yid] for yid in STMOVIES_YIDS])
NMMOVIES_IDS = np.array([item_mapper[yid] for yid in NMMOVIES_YIDS])
HWMOVIES_IDS = np.array([item_mapper[yid] for yid in HWMOVIES_YIDS])

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


recommenders = {}
graphs = {}
sessions = {}
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
        W_train = fit_collaborative_algorithm(collaborative_algorithm, URM_train)
    else:
        output_folder_path = find_best_configuration(basepath, recommender_name, add_underscore="NeuralFeatureCombiner" in recommender_name)

    additional_parameters = {}
    if recommender_class in [NeuralFeatureCombiner, NeuralFeatureCombinerProfile, NeuralFeatureCombinerProfileBPR]:

        graphs[recommender_name] = tf.Graph()

        config = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                          intra_op_parallelism_threads=0,
                                          inter_op_parallelism_threads=0)
        config.gpu_options.allow_growth = True
        sessions[recommender_name] = tf.compat.v1.Session(config=config, graph=graphs[recommender_name])

        additional_parameters['checkpoint_path'] = zipFile_path + "tf_train_{}".format(recommender_name) 
        additional_parameters['log_path'] = additional_parameters['checkpoint_path'] + os.sep + "logs" + os.sep
        additional_parameters['clear_session_after_train'] = False
        additional_parameters['graph'] = graphs[recommender_name]
        additional_parameters['session'] = sessions[recommender_name]

    recommender, _ = train_best_config(recommender_class, train, output_folder_path, dataset_validation=test,
                                icm_name=icm_name, W_train=W_train, additional_parameters=additional_parameters)
    recommenders[recommender_name] = recommender

del recommender



def show_item_neighbors(items=None, toshow=5):

    if items is None:
        items = ["1800080795", "1800381592"]

    for item in items:

        if isinstance(item, str):
            item = item_mapper[item]

        print(raw_item_features[inv_item_mapper[item]][0])

        for recommendername, recommender in recommenders.items():

            W_sparse = sim.cosine(sps.csr_matrix(recommender.get_embedded_ICM()), k=toshow+5).transpose().tocsc()
            W_sparse.setdiag(0)

            assert W_sparse.shape[0] == W_sparse.shape[1] and W_sparse.shape[0] == ICM_object.shape[0]

            order = np.argsort(-W_sparse.data[W_sparse.indptr[item]:W_sparse.indptr[item+1]])[:toshow]
            most_similar = W_sparse.indices[W_sparse.indptr[item]:W_sparse.indptr[item+1]][order]
            print("{}:".format(recommendername), ", ".join(
                raw_item_features[inv_item_mapper[i]][0] for i in most_similar.tolist()))
            for i in most_similar.tolist():
                print(raw_item_features[inv_item_mapper[i]][0])
                #print(raw_item_features[inv_item_mapper[i]][0], ", ".join(feature_name(f)
                #                    for f in ICM_object.indices[ICM_object.indptr[i]:ICM_object.indptr[i+1]]))

        print()



def cosine(a, b):
    if sps.issparse(a):
        numerator = a.dot(b.T).toarray()
        normf = sps.linalg.norm
    else:
        numerator = np.matmul(a, b.T)
        normf = np.linalg.norm
    denominator = np.outer(normf(a, axis=-1).flatten(), normf(b, axis=-1).flatten())
    return np.divide(numerator, denominator)


def calculate_contributions(recommender, features=None, type="deeplift", per_output=False):
    recommender_name = recommender.RECOMMENDER_NAME
    graph = graphs[recommender_name]
    session = sessions[recommender_name]
    with DeepExplain(graph=graph, session=session) as de:
        rfw = NeuralFeatureCombinerFW(URM_train, ICM_object, de, recommender)
        contributions = rfw.compute_weights(features=features, type=type, per_output=per_output)
    return contributions


def print_contributions(moviesname, recommendername, contributions, toshow=None, collapse=True):
    syn_str = 0 if _NO_SYNOPSIS else 1
    if collapse:
        first_layer = np.mean(contributions[0], axis=1)
    else:
        first_layer = contributions
    mask = first_layer != 0
    indices = np.arange(len(first_layer))[mask]
    features_to_show = np.argsort(-first_layer[mask])
    if toshow is not None:
        features_to_show = features_to_show[:toshow]
    features_to_show = indices[features_to_show]
    with open(_FW_FILENAME, "a") as file:
        maxf = np.max(first_layer)
        for f in features_to_show:
            print("{},{},{},{},{}".format(moviesname, recommendername, syn_str, 
                               feature_name(f), first_layer[f] / maxf), file=file)


def genre_analysis(genres=None):

    if genres is None:
        genres = [name for (name, lvl) in _GENRES if lvl > _GENRE_FILTER]

    movies_per_genre = {}
    for genre in genres:
        movies_per_genre[genre] = []
        for id in range(URM_train.shape[1]):
            if genre in raw_item_features[inv_item_mapper[id]][8]:
                movies_per_genre[genre].append(id)

    for recommendername, recommender in recommenders.items():
        data = []
        for idx, genre1 in enumerate(genres):
            items1 = recommender.get_embedding(ICM_object[np.array(movies_per_genre[genre1]), :])
            for genre2 in genres[idx:]:
                items2 = recommender.get_embedding(ICM_object[np.array(movies_per_genre[genre2]), :])
                data.append({
                    'genre1': genre1,
                    'genre2': genre2,
                    'similarity': np.mean(cosine(items1, items2)),
                })

        df = pd.DataFrame(data).pivot('genre2', 'genre1', 'similarity').reindex(genres)[genres]
        ax = sns.heatmap(df, square=True)
        ax.set_xticklabels(ax.get_xticklabels(), rotation_mode="anchor", horizontalalignment="right", verticalalignment="top", rotation=30)
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(zipFile_path + "genre_heatmap_{}.png".format(recommendername))
        plt.clf()



def neighbors_analysis(movies_ids, rfw, moviesname, toshow=20):

    print("---------------- {} ----------------".format(moviesname.upper()))

    if _NOISY:
        noisy_steps = range(0, 10, intperc_step)
    else:
        noisy_steps = [0]

    movies_noisy_features = {}
    np.random.seed(42)
    intperc_step = 2
    for intperc in range(0, 10, intperc_step):
        perc = float(intperc) / 10
        noisy_features = ICM_object[movies_ids, :].copy()
        for item in range(noisy_features.shape[0]):
            nfeatures = noisy_features.indptr[item+1] - noisy_features.indptr[item]
            todelete = np.random.choice(nfeatures, int(perc * nfeatures)) + noisy_features.indptr[item]
            noisy_features.data[todelete] = 0.
        noisy_features.eliminate_zeros()
        movies_noisy_features[intperc] = noisy_features

    movies_noisy_neighbors = {}
    for intperc in noisy_steps:

        movies_features = movies_noisy_features[intperc]
        movies_noisy_neighbors[intperc] = {}

        for recommendername, recommender in recommenders.items():

            embedded_movies = sps.csr_matrix(recommender.get_embedding(movies_features))
            assert embedded_movies.shape[0] == movies_features.shape[0]

            sim_mat = cosine(embedded_movies, sps.csr_matrix(recommender.get_embedded_ICM()))
            movies_noisy_neighbors[intperc][recommendername] = np.argsort(-sim_mat)[:, :toshow]

    contributions = {
        "ItemKNNCBF": [],
        "CFW_D": [],
        "NeuralFeatureCombiner": []
    }
    #noisy_contributions = []
    df_data = []
    for intperc in noisy_steps:

        perc = float(intperc) / 10
        movies_features = movies_noisy_features[intperc]

        #print("---------------------- Noise perc: {:.0f}% -----------------------".format(perc*100))

        for i, item in enumerate(movies_ids):

            for recommendername, recommender in recommenders.items():

                most_similar = movies_noisy_neighbors[intperc][recommendername][i, :]

                df_data.append({
                    "Recommender": recommendername,
                    "perc": perc,
                    "item": i,
                    "movies": np.in1d(most_similar, movies_ids).sum() / len(movies_ids) * 100
                })

                if recommendername in contributions.keys():
                    if isinstance(recommender, NeuralFeatureCombiner):
                        contributions[recommendername].append(
                            sps.csr_matrix(rfw.compute_weights(features=movies_features[i, :], type="deeplift", per_input=False))
                        )
                    else:
                        contributions[recommendername].append(recommender.get_embedding(movies_features[i, :]))

    for recommendername, contr in contributions.items():
        avg_contribution = np.array(sps.vstack(contr).mean(axis=0)).flatten()
        print_contributions(moviesname, recommendername, avg_contribution, collapse=False)
    
    #if len(noisy_contributions) > 0:
    #    avg_contribution = np.mean(np.vstack([np.mean(c[0], axis=1).flatten() for c in noisy_contributions]), axis=0)
    #    print_contributions(avg_contribution, toshow=20, collapse=False)
    #    print()

    df = pd.DataFrame(df_data)
    print(df.groupby(["Recommender", "perc"]).mean())
    ax = sns.boxplot(x="perc", y="movies", hue="Recommender", data=df)

    fig = ax.get_figure()
    fig.tight_layout()

    fig.savefig(zipFile_path + "noise_test_{}_{}.png".format(toshow, moviesname))

    plt.clf()


def find_actors(actors_names=None):

    if actors_names is None:
        actors_names = ['sylvester stallone', 'arnold schwarzenegger', 'pierce brosnan', 'sean connery',
                        'tom cruise', 'brad pitt', 'anthony hopkins', 'scarlett johansson', 'steven spielberg']

    ICM_object_csc = ICM_object.tocsc()

    for f in actors_names:
        fid = None
        try:
            print(f, directors_name_to_id[f])
            fid = directors_name_to_id[f]
        except Exception as e:
            print(f, "not found in directors")
            pass
        try:
            print(f, actors_name_to_id[f])
            fid = actors_name_to_id[f]
        except Exception as e:
            print(f, "not found in actors")
            pass

        if fid is not None:
            mids = [int(feature_mapper[k]) for k in fm_keys if fid in k]
            for mid in mids:
                items = ICM_object_csc.indices[ICM_object_csc.indptr[mid]:ICM_object_csc.indptr[mid+1]]
                for item in items.tolist():
                    try:
                        print(item, raw_item_features[inv_item_mapper[item]])
                    except Exception as e:
                        print(item, "not found in items")



if __name__ == "__main__":
    show_item_neighbors()
    genre_analysis()
    toshow = 20 
    recommender_name = "NeuralFeatureCombiner"
    with DeepExplain(graph=graphs[recommender_name], session=sessions[recommender_name]) as de:
        rfw = NeuralFeatureCombinerFW(URM_train, ICM_object, de, recommenders[recommender_name])
        with open(_FW_FILENAME, "w") as file:
            print("movies,recommender,synopsis,feature,weight", file=file)
        neighbors_analysis(JBMOVIES_IDS, rfw, "james-bond", toshow=toshow)
        neighbors_analysis(SWMOVIES_IDS, rfw, "star-wars", toshow=toshow)
        neighbors_analysis(STMOVIES_IDS, rfw, "star-trek", toshow=toshow)
        neighbors_analysis(NMMOVIES_IDS, rfw, "nightmare", toshow=toshow)
        neighbors_analysis(HWMOVIES_IDS, rfw, "halloween", toshow=toshow)


