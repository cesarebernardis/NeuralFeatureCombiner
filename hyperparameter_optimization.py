import numpy as np
import os
import pickle

from RecSysFramework.Recommender.MatrixFactorization import FBSM, LCE, DCT, BPRMF_AFM
from RecSysFramework.Recommender import FactorizationMachine
from RecSysFramework.Recommender.KNN import CFW_D, ItemKNNCBF, ItemKNNCF
from RecSysFramework.Recommender.GraphBased import HP3
from RecSysFramework.Recommender.DeepLearning import NeuralFeatureCombiner_OptimizerMask as NeuralFeatureCombiner
from RecSysFramework.Recommender.DeepLearning import NeuralFeatureCombinerProfile_OptimizerMask as NeuralFeatureCombinerProfile
from RecSysFramework.Recommender.DeepLearning import NeuralFeatureCombinerProfileBPR_OptimizerMask as NeuralFeatureCombinerProfileBPR
from RecSysFramework.Recommender.DeepLearning import NeuralFeatureCombinerProfileCE_OptimizerMask as NeuralFeatureCombinerProfileCE
from RecSysFramework.Recommender.DeepLearning import WideAndDeep_OptimizerMask as WideAndDeep

from RecSysFramework.Recommender.DataIO import DataIO

from RecSysFramework.ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from RecSysFramework.ParameterTuning.Utils import run_parameter_search
from RecSysFramework.ExperimentalConfig import EXPERIMENTAL_CONFIG


def train_and_save(algorithm, dataset_train, basepath, icm_name=None, W_train=None, outputpath=None):

    if outputpath is None:
        outputpath = basepath
    os.makedirs(outputpath, exist_ok=True)

    dataIO = DataIO(folder_path=basepath)
    data_dict = dataIO.load_data(file_name=algorithm.RECOMMENDER_NAME + "_metadata")

    cpa = [dataset_train.get_URM()]
    if icm_name is not None:
        cpa.append(dataset_train.get_ICM(icm_name))
    if W_train is not None:
        cpa.append(W_train)

    recommender = algorithm(*cpa)
    recommender.fit(**data_dict["hyperparameters_best"])
    recommender.save_model(outputpath, file_name="{}_best_model".format(algorithm.RECOMMENDER_NAME))


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


if __name__ == "__main__":

    _metric_to_optimize = "NDCG"
    _cutoff_to_optimize = 10
    _n_cases = 50
    _n_random_starts = 15

    optimize_collaborative = True
    optimize_coldstart = True
    optimize_coldstart_profile = True
    optimize_coldstart_sim = True

    resume_from_saved = True

    for dataset_config in EXPERIMENTAL_CONFIG['datasets']:

        datareader = dataset_config['datareader']()
        postprocessings = dataset_config['postprocessings']
        subsampling_pp = dataset_config['subsampling_postprocessings']

        splitter = EXPERIMENTAL_CONFIG['cold_split']
        collaborative_splitter = EXPERIMENTAL_CONFIG['warm_split']

        basepath = splitter.get_complete_default_save_folder_path(datareader, postprocessings=postprocessings)

        dataset_train, dataset_validation = splitter.load_split(datareader,
                                postprocessings=postprocessings, save_folder_path=basepath)

        if subsampling_pp is not None:
            complete_basepath = basepath + os.sep.join(pp.get_name() for pp in subsampling_pp) + os.sep
            dataset_subtrain, dataset_subvalidation = splitter.load_split(datareader,
                            postprocessings=postprocessings + subsampling_pp, save_folder_path=complete_basepath)
        else:
            dataset_subtrain = dataset_train
            dataset_subvalidation = dataset_validation

        cold_basepath = basepath + splitter.get_name() + os.sep
        collaborative_basepath = cold_basepath + collaborative_splitter.get_name() + os.sep

        interactions = np.ediff1d(dataset_validation.get_URM().tocsc().indptr)
        ignore_items = np.arange(dataset_validation.n_items)[interactions == 0]

        interactions = np.ediff1d(dataset_subvalidation.get_URM().tocsc().indptr)
        ignore_items_subsample = np.arange(dataset_subvalidation.n_items)[interactions == 0]

        if optimize_collaborative:

            collaborative_dataset_train, collaborative_dataset_validation = collaborative_splitter.load_split(
                    datareader, postprocessings=postprocessings, save_folder_path=cold_basepath)

            for algorithm in EXPERIMENTAL_CONFIG["collaborative_algorithms"]:

                output_folder_path = collaborative_basepath + algorithm.RECOMMENDER_NAME + os.sep

                run_parameter_search(
                    algorithm, collaborative_splitter.get_name(),
                    collaborative_dataset_train, collaborative_dataset_validation,
                    output_folder_path=output_folder_path, metric_to_optimize=_metric_to_optimize,
                    cutoff_to_optimize=_cutoff_to_optimize, resume_from_saved=resume_from_saved,
                    n_cases=_n_cases, n_random_starts=_n_random_starts, save_model="no"
                )

                train_and_save(algorithm, dataset_train, output_folder_path,
                               outputpath=cold_basepath + algorithm.RECOMMENDER_NAME + os.sep)

            del collaborative_dataset_train
            del collaborative_dataset_validation

        to_optimize = []
        if optimize_coldstart:
            to_optimize += [
                {'class': ItemKNNCBF},
                {'class': FBSM},
                {'class': LCE},
                {'class': DCT},
                {'class': FactorizationMachine},
                {'class': BPRMF_AFM}
            ]
            for l in [1, 2, 3]:
                to_optimize.append({'class': WideAndDeep, 'layers': l})

        if optimize_coldstart_profile:
            for el in [2, 3]:
                for dl in [0, 1]:
                    to_optimize.extend([
                        {'class': NeuralFeatureCombinerProfile, 'encoder_layers': el, 'decoder_layers': dl, 'apply_subsample': True},
                        {'class': NeuralFeatureCombinerProfileBPR, 'encoder_layers': el, 'decoder_layers': dl, 'apply_subsample': True},
                        {'class': NeuralFeatureCombinerProfileCE, 'encoder_layers': el, 'decoder_layers': dl, 'apply_subsample': True},
                    ])

        for specs in to_optimize:

            algorithm, recommender_name, kwargs = parse_specs(specs)

            datatrain = dataset_train
            dataval = dataset_validation
            ignoreitems = ignore_items
            if 'apply_subsample' in kwargs.keys():
                if kwargs['apply_subsample']:
                    datatrain = dataset_subtrain
                    dataval = dataset_subvalidation
                    ignoreitems = ignore_items_subsample
                del kwargs['apply_subsample']

            if algorithm is DCT:
                ICM_name = None
                dataIO = DataIO(folder_path=cold_basepath + ItemKNNCBF.RECOMMENDER_NAME + os.sep)
                data_dict = dataIO.load_data(file_name=ItemKNNCBF.RECOMMENDER_NAME + "_metadata")
                recommender = ItemKNNCBF(datatrain.get_URM(), datatrain.get_ICM())
                recommender.fit(**data_dict["hyperparameters_best"])
                kwargs["W_train"] = recommender.get_W_sparse()
                del recommender
            else:
                ICM_name = "ICM_all"

            output_folder_path = cold_basepath + recommender_name + os.sep

            run_parameter_search(
                algorithm, splitter.get_name(), datatrain, dataval, ICM_name=ICM_name,
                output_folder_path=output_folder_path, ignore_items_validation=ignoreitems,
                metric_to_optimize=_metric_to_optimize, cutoff_to_optimize=_cutoff_to_optimize,
                resume_from_saved=resume_from_saved,
                n_cases=_n_cases, n_random_starts=_n_random_starts, save_model="no", **kwargs
            )

        if optimize_coldstart_sim:

            for collaborative_algorithm in EXPERIMENTAL_CONFIG["collaborative_algorithms"]:

                algo_basepath = cold_basepath + collaborative_algorithm.RECOMMENDER_NAME + os.sep
                recommender = collaborative_algorithm(dataset_train.get_URM())
                model_filename = "{}_best_model".format(collaborative_algorithm.RECOMMENDER_NAME)
                if not os.path.exists(algo_basepath + os.sep + model_filename):
                    train_and_save(collaborative_algorithm, dataset_train,
                                   collaborative_basepath + collaborative_algorithm.RECOMMENDER_NAME + os.sep,
                                   outputpath=cold_basepath + collaborative_algorithm.RECOMMENDER_NAME + os.sep)
                recommender.load_model(algo_basepath, file_name=model_filename)
                W_train = recommender.get_W_sparse()
                del recommender

                to_optimize = [
                    {'class': CFW_D},
                    {'class': HP3}
                ]

                for el in [2, 3]:
                    for dl in [0, 1]:
                        to_optimize.append(
                            {'class': NeuralFeatureCombiner, 'encoder_layers': el, 'decoder_layers': dl}
                        )

                for specs in to_optimize:

                    algorithm, recommender_name, kwargs = parse_specs(specs)
                    output_folder_path = algo_basepath + recommender_name + os.sep

                    run_parameter_search(
                        algorithm, splitter.get_name(), dataset_train, dataset_validation,
                        W_train=W_train, output_folder_path=output_folder_path, ICM_name="ICM_all",
                        metric_to_optimize=_metric_to_optimize, cutoff_to_optimize=_cutoff_to_optimize,
                        resume_from_saved=resume_from_saved, n_cases=_n_cases, n_random_starts=_n_random_starts,
                        save_model="no", ignore_items_validation=ignore_items, **kwargs
                    )

                del W_train

