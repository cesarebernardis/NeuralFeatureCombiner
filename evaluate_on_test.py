import numpy as np
import pickle
import os
import time
import glob
import argparse

from RecSysFramework.Recommender.MatrixFactorization import FBSM, LCE, DCT, BPRMF_AFM
from RecSysFramework.Recommender import FactorizationMachine
from RecSysFramework.Recommender.SLIM.ElasticNet import SLIM
from RecSysFramework.Recommender.KNN import CFW_D, ItemKNNCBF, ItemKNNCF, EASE_R
from RecSysFramework.Recommender.GraphBased import HP3, RP3beta
from RecSysFramework.Recommender.DeepLearning import NeuralFeatureCombiner_OptimizerMask as NeuralFeatureCombiner
from RecSysFramework.Recommender.DeepLearning import NeuralFeatureCombinerProfile_OptimizerMask as NeuralFeatureCombinerProfile
from RecSysFramework.Recommender.DeepLearning import NeuralFeatureCombinerProfileBPR_OptimizerMask as NeuralFeatureCombinerProfileBPR
from RecSysFramework.Recommender.DeepLearning import NeuralFeatureCombinerProfileCE_OptimizerMask as NeuralFeatureCombinerProfileCE
from RecSysFramework.Recommender.DeepLearning import WideAndDeep_OptimizerMask as WideAndDeep

from RecSysFramework.Recommender.DataIO import DataIO

from RecSysFramework.Evaluation import EvaluatorHoldout

from RecSysFramework.DataManager.Reader import BookCrossingReader, AmazonGamesReader

from RecSysFramework.ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from RecSysFramework.ParameterTuning.Utils import run_parameter_search
from RecSysFramework.ExperimentalConfig import EXPERIMENTAL_CONFIG



_DEFAULT_RESULT = {}
for cutoff in EXPERIMENTAL_CONFIG['cutoffs']:
    _DEFAULT_RESULT[cutoff] = {}
    for metric in EXPERIMENTAL_CONFIG['recap_metrics']:
        _DEFAULT_RESULT[cutoff][metric] = 0.


def train_best_config(algorithm, dataset_train, basepath, dataset_validation=None,
                      icm_name=None, W_train=None, save=False, additional_parameters=None):

    dataIO = DataIO(folder_path=basepath)
    data_dict = dataIO.load_data(file_name=algorithm.RECOMMENDER_NAME + "_metadata")

    if isinstance(dataset_train, list):
        posargs = []
        for i, train in enumerate(dataset_train):
            urm = train.get_URM()
            if dataset_validation is not None:
                urm += dataset_validation[i].get_URM()
            cpa = [urm]
            if icm_name is not None:
                cpa.append(train.get_ICM(icm_name))
            if W_train is not None:
                cpa.append(W_train[i])
            posargs.append(SearchInputRecommenderArgs(CONSTRUCTOR_POSITIONAL_ARGS=cpa))
        start_time = time.time()
        recommender = Recommender_k_Fold_Wrapper(algorithm, posargs)
    else:
        urm = dataset_train.get_URM()
        if dataset_validation is not None:
            urm += dataset_validation.get_URM()
        cpa = [urm]
        if icm_name is not None:
            cpa.append(dataset_train.get_ICM(icm_name))
        if W_train is not None:
            cpa.append(W_train)
        start_time = time.time()
        recommender = algorithm(*cpa)

    best_parameters = data_dict["hyperparameters_best"]
    if additional_parameters is not None:
        for k, v in additional_parameters.items():
            best_parameters[k] = v

    recommender.fit(**best_parameters)
    end_time = time.time()

    if save:
        recommender.save_model(basepath, file_name="{}_best_model".format(algorithm.RECOMMENDER_NAME))

    return recommender, end_time - start_time


def is_fold_evaluated(basepath, fold_splitter, fold, force_evaluation=False):
    if force_evaluation:
        return False
    filename = basepath + "evaluated_folds.pkl"
    fold_name = fold_splitter.get_name()
    if os.path.exists(filename):
        with open(filename, "rb") as file:
            folds = pickle.load(file)
            if fold_name in folds.keys():
                return fold in folds[fold_name]
    return False


def set_fold_evaluated(basepath, fold_splitter, fold):
    filename = basepath + "evaluated_folds.pkl"
    fold_name = fold_splitter.get_name()
    if os.path.exists(filename):
        with open(filename, "rb") as file:
            folds = pickle.load(file)
    else:
        folds = {}
    if fold_name not in folds.keys():
        folds[fold_name] = []
    folds[fold_name].append(fold)
    with open(filename, "wb") as file:
        pickle.dump(folds, file)


def parse_specs(specs):
    kwargs = {}
    algorithm = None
    for k, v in specs.items():
        if k == "class":
            algorithm = v
        else:
            kwargs[k] = v

    recommender_name = algorithm.RECOMMENDER_NAME
    if "layers" in kwargs.keys():
        recommender_name += "_{}".format(kwargs["layers"])
    elif "encoder_layers" in kwargs.keys() and "decoder_layers" in kwargs.keys():
        recommender_name += "_{}_{}".format(kwargs["encoder_layers"], kwargs["decoder_layers"])

    return algorithm, recommender_name, kwargs


def find_best_configuration(basepath, recommender_name, add_underscore=False):
    best_result_validation = -1
    best_output_folder_path = None
    if add_underscore:
        add_str = "_*"
    else:
        add_str = "*"
    for output_folder_path in glob.glob(basepath + "**" + os.sep + recommender_name + add_str + os.sep, recursive=True):
        dataIO = DataIO(folder_path=output_folder_path)
        data_dict = dataIO.load_data(file_name=recommender_name + "_metadata")
        tmp_best_result = data_dict["result_on_validation_best"]["NDCG"]
        if tmp_best_result > best_result_validation:
            best_output_folder_path = output_folder_path
            best_result_validation = tmp_best_result
    return best_output_folder_path


def algorithm_name_to_class(name):
    for algorithm in EXPERIMENTAL_CONFIG["collaborative_algorithms"]:
        if algorithm.RECOMMENDER_NAME == name:
            return algorithm


def write_results(output_folder_path, results, dataset_name, fold_name, fold,
                  recommender_name, train_time, collaborative_algorithm=""):
    results_filename = output_folder_path + "test_optimal_results.txt"
    with open(results_filename, "a") as file:
        for cutoff in EXPERIMENTAL_CONFIG['cutoffs']:
            print("{},{},{},{},{},{},{},{}".format(
                dataset_name, fold_name, fold, recommender_name, collaborative_algorithm, cutoff,
                ','.join("{:.5f}".format(results[cutoff][m]) for m in
                         EXPERIMENTAL_CONFIG['recap_metrics']), train_time), file=file)



def run_complete_evaluation(dataset_config, fold_splitter, fold, force_evaluation=False,
                            run_all_similarities=False, run_collaborative=False, selected_algorithms=False):

    datareader = dataset_config['datareader']()
    postprocessings = dataset_config['postprocessings']
    dataset_name = datareader.get_dataset_name()
    fold_name = fold_splitter.get_name()

    splitter = EXPERIMENTAL_CONFIG['cold_split']
    collaborative_splitter = EXPERIMENTAL_CONFIG['warm_split']

    dataset_train, dataset_test = fold_splitter.load_split(datareader,
                                                           postprocessings=postprocessings,
                                                           filename_suffix="_{}".format(fold))

    #ignore items that are not in test
    interactions = np.ediff1d(dataset_test.get_URM().tocsc().indptr)
    ignore_items = np.arange(dataset_test.n_items)[interactions == 0]

    #ignore cold start users
    interactions = np.ediff1d(dataset_train.get_URM().tocsr().indptr)
    ignore_users = np.arange(dataset_train.n_users)[interactions == 0]

    basepath = splitter.get_complete_default_save_folder_path(datareader, postprocessings=postprocessings)
    cold_basepath = basepath + splitter.get_name() + os.sep
    collaborative_basepath = cold_basepath + collaborative_splitter.get_name() + os.sep

    evaluator = EvaluatorHoldout(cutoff_list=EXPERIMENTAL_CONFIG['cutoffs'],
                                 metrics_list=EXPERIMENTAL_CONFIG['recap_metrics'])

    evaluator.global_setup(dataset_test.get_URM(), ignore_items=ignore_items, ignore_users=ignore_users)

    if run_collaborative:

        for algorithm in EXPERIMENTAL_CONFIG["collaborative_algorithms"]:

            recommender_name = algorithm.RECOMMENDER_NAME
            output_folder_path = collaborative_basepath + recommender_name + os.sep

            if is_fold_evaluated(output_folder_path, fold_splitter, fold, force_evaluation=force_evaluation):
                continue

            recommender, train_time = train_best_config(algorithm, dataset_train, output_folder_path)

            if "hybrid" in fold_name:
                metrics_handler = evaluator.evaluateRecommender(recommender)
                results = metrics_handler.get_results_dictionary(use_metric_name=True)
            else:
                # In case we only need to compute the time, e.g. in cold-start
                results = _DEFAULT_RESULT

            write_results(output_folder_path, results, dataset_name, fold_name, fold, recommender_name, train_time)
            set_fold_evaluated(output_folder_path, fold_splitter, fold)

    to_optimize = [
        {'class': ItemKNNCBF},
        {'class': FBSM},
        {'class': DCT}
    ]

    if not selected_algorithms:
        to_optimize += [
            {'class': LCE},
            {'class': BPRMF_AFM},
            {'class': FactorizationMachine},
            {'class': WideAndDeep, 'find_best_config': True},
            {'class': NeuralFeatureCombinerProfile, 'find_best_config': True},
            {'class': NeuralFeatureCombinerProfileBPR, 'find_best_config': True},
            {'class': NeuralFeatureCombinerProfileCE, 'find_best_config': True},
        ]

    for specs in to_optimize:

        algorithm, recommender_name, kwargs = parse_specs(specs)

        W_train = None
        ICM_name = "ICM_all"
        if algorithm is DCT:
            ICM_name = None
            dataIO = DataIO(folder_path=cold_basepath + ItemKNNCBF.RECOMMENDER_NAME + os.sep)
            data_dict = dataIO.load_data(file_name=ItemKNNCBF.RECOMMENDER_NAME + "_metadata")
            recommender = ItemKNNCBF(dataset_train.get_URM(), dataset_train.get_ICM())
            recommender.fit(**data_dict["hyperparameters_best"])
            W_train = recommender.get_W_sparse()
            del recommender

        if 'find_best_config' in kwargs.keys() and kwargs['find_best_config']:
            output_folder_path = find_best_configuration(basepath, recommender_name, add_underscore=True)
        else:
            output_folder_path = cold_basepath + recommender_name + os.sep

        if is_fold_evaluated(output_folder_path, fold_splitter, fold, force_evaluation=force_evaluation):
            continue

        recommender, train_time = train_best_config(algorithm, dataset_train, output_folder_path,
                                                    icm_name=ICM_name, W_train=W_train)

        metrics_handler = evaluator.evaluateRecommender(recommender)
        results = metrics_handler.get_results_dictionary(use_metric_name=True)

        write_results(output_folder_path, results, dataset_name, fold_name, fold, recommender_name, train_time)
        set_fold_evaluated(output_folder_path, fold_splitter, fold)

        cs_op = getattr(recommender, "clear_session", None)
        if cs_op is not None and callable(cs_op):
            recommender.clear_session()
        del recommender

    to_optimize = [{'class': HP3}]
    if not selected_algorithms:
        to_optimize.append({'class': CFW_D})

    if run_all_similarities:
        collaborative_algorithms = EXPERIMENTAL_CONFIG["collaborative_algorithms"]
        for el in [1, 2, 3]:
            for dl in [0, 1]:
                to_optimize.append(
                    {'class': NeuralFeatureCombiner, 'encoder_layers': el, 'decoder_layers': dl}
                )
    else:
        collaborative_algorithms = [None]
        to_optimize.append({'class': NeuralFeatureCombiner})

    def fit_collaborative_algorithm(_collaborative_algorithm, _urm):
        algo_basepath = cold_basepath + _collaborative_algorithm.RECOMMENDER_NAME + os.sep
        dataIO = DataIO(folder_path=algo_basepath)
        data_dict = dataIO.load_data(file_name=_collaborative_algorithm.RECOMMENDER_NAME + "_metadata")
        recommender = _collaborative_algorithm(_urm)
        recommender.fit(**data_dict["hyperparameters_best"])
        W_train = recommender.get_W_sparse()
        del recommender
        return W_train

    for collaborative_algorithm in collaborative_algorithms:

        if collaborative_algorithm is not None:
            W_train = fit_collaborative_algorithm(collaborative_algorithm, dataset_train.get_URM())

        for specs in to_optimize:

            algorithm, recommender_name, kwargs = parse_specs(specs)

            if collaborative_algorithm is None:
                output_folder_path = find_best_configuration(basepath, recommender_name,
                                                    add_underscore="NeuralFeatureCombiner" in recommender_name)
                collaborative_recommender_name = output_folder_path.split(os.sep)[-3]
                ca_class = algorithm_name_to_class(collaborative_recommender_name)
                W_train = fit_collaborative_algorithm(ca_class, dataset_train.get_URM())
            else:
                collaborative_recommender_name = collaborative_algorithm.RECOMMENDER_NAME
                output_folder_path = cold_basepath + collaborative_recommender_name + os.sep + \
                                     recommender_name + os.sep

            if is_fold_evaluated(output_folder_path, fold_splitter, fold, force_evaluation=force_evaluation):
                continue

            recommender, train_time = train_best_config(algorithm, dataset_train, output_folder_path,
                                                        W_train=W_train, icm_name="ICM_all")

            metrics_handler = evaluator.evaluateRecommender(recommender)
            results = metrics_handler.get_results_dictionary(use_metric_name=True)

            write_results(output_folder_path, results, dataset_name, fold_name, fold, recommender_name,
                          train_time, collaborative_algorithm=collaborative_recommender_name)
            set_fold_evaluated(output_folder_path, fold_splitter, fold)

            cs_op = getattr(recommender, "clear_session", None)
            if cs_op is not None and callable(cs_op):
                recommender.clear_session()
            del recommender

        del W_train



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Runs top-n recommendation tests')
    parser.add_argument('--cold', dest="cold", const=True, default=False, nargs="?",
                        help='If specified, runs the tests in the cold-start scenario')
    parser.add_argument('--rampup', dest="rampup", const=True, default=False, nargs="?",
                        help='If specified, runs the tests in the ramp-up scenario')
    parser.add_argument('--hybrid', dest="hybrid", const=True, default=False, nargs="?",
                        help='If specified, runs the tests in the hybrid scenario')
    parser.add_argument('--force', dest="force", const=True, default=False, nargs="?",
                        help='If specified, runs the tests without checking if results are already available')

    arguments = parser.parse_args()
    
    # Automatically saves the results and does not run the experiments twice, if the results are found
    force_evaluation = arguments.force
    dataset_selection = [BookCrossingReader, AmazonGamesReader]

    if arguments.cold:
        for dataset_config in EXPERIMENTAL_CONFIG['datasets']:
            for fold in range(EXPERIMENTAL_CONFIG['n_folds']):
                run_complete_evaluation(dataset_config, EXPERIMENTAL_CONFIG["cold_split"], fold,
                            force_evaluation=force_evaluation, selected_algorithms=False, run_collaborative=True)

    if arguments.rampup:
        for dataset_config in EXPERIMENTAL_CONFIG['datasets']:
            if dataset_config['datareader'] in dataset_selection:
                for fold in range(EXPERIMENTAL_CONFIG['n_folds']):
                    for fold_splitter in EXPERIMENTAL_CONFIG['cold_split_perc'][1:]:
                        run_complete_evaluation(dataset_config, fold_splitter, fold, force_evaluation=force_evaluation,
                                                selected_algorithms=True, run_collaborative=False)

    if arguments.hybrid:
        for dataset_config in EXPERIMENTAL_CONFIG['datasets']:
            if dataset_config['datareader'] in dataset_selection:
                for fold in range(EXPERIMENTAL_CONFIG['n_folds']):
                    for fold_splitter in EXPERIMENTAL_CONFIG['hybrid_split_perc']:
                        run_complete_evaluation(dataset_config, fold_splitter, fold, force_evaluation=force_evaluation,
                                                selected_algorithms=True, run_collaborative=True)

