import os
import traceback

from skopt.space import Real, Integer, Categorical

from RecSysFramework.Recommender.NonPersonalized import TopPop, Random, GlobalEffects
from RecSysFramework.Recommender.KNN import UserKNNCF
from RecSysFramework.Recommender.KNN import ItemKNNCF, ItemKNNCBF
from RecSysFramework.Recommender.KNN import EASE_R
from RecSysFramework.Recommender.KNN import CFW_D, CFW_DVV
from RecSysFramework.Recommender.SLIM.BPR import SLIM as SLIM_BPR
from RecSysFramework.Recommender.SLIM.ElasticNet import SLIM as SLIM_ElasticNet
from RecSysFramework.Recommender.GraphBased import P3alpha, RP3beta, HP3

from RecSysFramework.Recommender import FactorizationMachine
from RecSysFramework.Recommender import FactorizationMachineProfile

from RecSysFramework.Recommender.KNN import ItemKNNCFCBFHybrid

from RecSysFramework.Recommender.DeepLearning import WideAndDeep_OptimizerMask as WideAndDeep
from RecSysFramework.Recommender.DeepLearning import WideAndDeepProfile_OptimizerMask as WideAndDeepProfile
from RecSysFramework.Recommender.DeepLearning import NeuralFeatureCombiner_OptimizerMask as NeuralFeatureCombiner
from RecSysFramework.Recommender.DeepLearning import NeuralFeatureCombinerProfile_OptimizerMask as NeuralFeatureCombinerProfile
from RecSysFramework.Recommender.DeepLearning import NeuralFeatureCombinerProfileCE_OptimizerMask as NeuralFeatureCombinerProfileCE
from RecSysFramework.Recommender.DeepLearning import NeuralFeatureCombinerProfileBPR_OptimizerMask as NeuralFeatureCombinerProfileBPR
from RecSysFramework.Recommender.DeepLearning import NeuralFeatureCombinerProfileBPRMax_OptimizerMask as NeuralFeatureCombinerProfileBPRMax
from RecSysFramework.Recommender.DeepLearning import NeuralFeatureCombinerLight_OptimizerMask as NeuralFeatureCombinerLight
from RecSysFramework.Recommender.DeepLearning import NeuralFeatureCombinerLightProfile_OptimizerMask as NeuralFeatureCombinerLightProfile

from RecSysFramework.Recommender.MatrixFactorization import BPRMF, BPRMF_AFM, FunkSVD, AsySVD
from RecSysFramework.Recommender.MatrixFactorization import PureSVD, PureSVDSimilarity
from RecSysFramework.Recommender.MatrixFactorization import IALS
from RecSysFramework.Recommender.MatrixFactorization import NMF
from RecSysFramework.Recommender.MatrixFactorization import LCE, DCT, FBSM

from RecSysFramework.Utils import EarlyStoppingModel

from RecSysFramework.ParameterTuning import SearchBayesianSkopt
from RecSysFramework.ParameterTuning import SearchSingleCase
from RecSysFramework.ParameterTuning import Recommender_k_Fold_Wrapper, Evaluator_k_Fold_Wrapper
from RecSysFramework.ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs

from RecSysFramework.Evaluation import EvaluatorHoldout


def _build_ranges_dictionary(ranges, fixed_params):
    params = {}
    for key, value in ranges.items():
        if key not in fixed_params.keys():
            params[key] = value
    return params


def _knn_similarity_helper(similarity_type):
    hyperparameters_range_dictionary = {}
    hyperparameters_range_dictionary["topK"] = Integer(50, 500, prior='log-uniform')
    hyperparameters_range_dictionary["shrink"] = Integer(1, 1000, prior='log-uniform')

    if similarity_type == "asymmetric":
        hyperparameters_range_dictionary["alpha"] = Real(low=0, high=2, prior='uniform')

    elif similarity_type == "tversky":
        hyperparameters_range_dictionary["alpha"] = Real(low=0, high=2, prior='uniform')
        hyperparameters_range_dictionary["beta"] = Real(low=0, high=2, prior='uniform')

    hyperparameters_range_dictionary["feature_weighting"] = Categorical(["none", "BM25", "TF-IDF"])

    return hyperparameters_range_dictionary


def run_parameter_search(recommender_class, split_name, dataset_train, dataset_validation, dataset_test=None,
                         URM_name="URM_all", ICM_name=None, output_folder_path=None, save_model="no",
                         metric_to_optimize="Recall", cutoff_to_optimize=10, n_cases=50, n_random_starts=15,
                         resume_from_saved=True, fixed_positional_args=None, fixed_keyword_args=None,
                         URM_train_last_test=None, **kwargs):

    if fixed_keyword_args is None:
        fixed_keyword_args = {}

    if fixed_positional_args is None:
        fixed_positional_args = []

    is_earlystopping = False
    if issubclass(recommender_class, EarlyStoppingModel):
        is_earlystopping = True

    if isinstance(dataset_train, list):
        is_crossvalidation = True
        assert len(dataset_train) == len(dataset_validation), \
            "Train and Validation dataset lists have different lengths"
        URM_train = [dataset.get_URM(URM_name) for dataset in dataset_train]
        URM_validation = [dataset.get_URM(URM_name) for dataset in dataset_validation]
        if ICM_name is not None:
            ICM_object = [dataset.get_ICM(ICM_name) for dataset in dataset_train]
        if dataset_test is not None:
            assert len(dataset_train) == len(dataset_test), \
                "Train and Test dataset lists have different lengths"
            URM_test = [dataset.get_URM(URM_name) for dataset in dataset_test]
        else:
            URM_test = None
    else:
        is_crossvalidation = False
        URM_train = dataset_train.get_URM(URM_name)
        URM_validation = dataset_validation.get_URM(URM_name)
        if ICM_name is not None:
            ICM_object = dataset_train.get_ICM(ICM_name)
        if dataset_test is not None:
            URM_test = dataset_test.get_URM(URM_name)
        else:
            URM_test = None

    constructor_positional_args = []
    if is_crossvalidation:
        constructor_positional_args.append(recommender_class)
        constructor_positional_args.append([])
        for i, urm in enumerate(URM_train):
            posargs = [urm]
            if ICM_name is not None:
                posargs.append(ICM_object[i])
            if "W_train" in kwargs:
                posargs.append(kwargs["W_train"][i])
            constructor_positional_args[1].append(
                SearchInputRecommenderArgs(
                    CONSTRUCTOR_POSITIONAL_ARGS=posargs
                )
            )

        evaluator_validation = []
        for i, urm in enumerate(URM_validation):
            evaluator_validation.append(EvaluatorHoldout(cutoff_list=cutoff_to_optimize,
                                                         metrics_list=metric_to_optimize))
            ignore_items = None
            if "ignore_items_validation" in kwargs:
                ignore_items = kwargs["ignore_items_validation"][i]
                #print("Ignoring {} items in fold {} of validation".format(len(ignore_items), i))
            ignore_users = None
            if "ignore_users_validation" in kwargs:
                ignore_users = kwargs["ignore_users_validation"][i]
                #print("Ignoring {} users in fold {} of validation".format(len(ignore_users), i))
            evaluator_validation[-1].global_setup(urm, ignore_users=ignore_users, ignore_items=ignore_items)

        evaluator_validation_earlystopping = evaluator_validation
        evaluator_validation = Evaluator_k_Fold_Wrapper(evaluator_validation)

    else:
        constructor_positional_args = [URM_train]
        if ICM_name is not None:
            constructor_positional_args.append(ICM_object)
        if "W_train" in kwargs:
            constructor_positional_args.append(kwargs["W_train"])
        evaluator_validation = EvaluatorHoldout(cutoff_list=cutoff_to_optimize,
                                                metrics_list=metric_to_optimize)
        ignore_items = None
        if "ignore_items_validation" in kwargs:
            ignore_items = kwargs["ignore_items_validation"]
            #print("Ignoring {} items in validation".format(len(ignore_items)))
        ignore_users = None
        if "ignore_users_validation" in kwargs:
            ignore_users = kwargs["ignore_users_validation"]
            #print("Ignoring {} users in validation".format(len(ignore_users)))
        evaluator_validation.global_setup(URM_validation, ignore_users=ignore_users, ignore_items=ignore_items)
        evaluator_validation_earlystopping = evaluator_validation

    if URM_test is not None:
        if is_crossvalidation:
            evaluator_test = []
            for i, urm in enumerate(URM_test):
                evaluator_test.append(EvaluatorHoldout(cutoff_list=cutoff_to_optimize,
                                                       metrics_list=metric_to_optimize))
                ignore_items = None
                if "ignore_items_test" in kwargs:
                    ignore_items = kwargs["ignore_items_test"][i]
                    #print("Ignoring {} items in fold {} of test".format(len(ignore_items), i))
                ignore_users = None
                if "ignore_users_test" in kwargs:
                    ignore_users = kwargs["ignore_users_test"][i]
                    #print("Ignoring {} users in fold {} of test".format(len(ignore_users), i))
                evaluator_test[-1].global_setup(urm, ignore_users=ignore_users, ignore_items=ignore_items)
            evaluator_test = Evaluator_k_Fold_Wrapper(evaluator_test)
        else:
            evaluator_test = EvaluatorHoldout(cutoff_list=cutoff_to_optimize,
                                              metrics_list=metric_to_optimize)
            ignore_items = None
            if "ignore_items_test" in kwargs:
                ignore_items = kwargs["ignore_items_test"]
                #print("Ignoring {} items in test".format(len(ignore_items)))
            ignore_users = None
            if "ignore_users_test" in kwargs:
                ignore_users = kwargs["ignore_users_test"]
                #print("Ignoring {} users in test".format(len(ignore_users)))
            evaluator_test.global_setup(URM_test, ignore_users=ignore_users, ignore_items=ignore_items)
    else:
        evaluator_test = None

    if URM_train_last_test is not None:
        URM_train_last_test = URM_train_last_test.copy()

    if output_folder_path is None:
        output_folder_path = dataset_train.get_complete_folder() + os.sep + \
                             split_name + os.sep + \
                             recommender_class.RECOMMENDER_NAME + os.sep

    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    try:

        output_file_name_root = recommender_class.RECOMMENDER_NAME

        if recommender_class in [TopPop, GlobalEffects, Random]:
            """
            TopPop, GlobalEffects and Random have no parameters therefore only one evaluation is needed
            """

            parameterSearch = SearchSingleCase(recommender_class,
                                               evaluator_validation=evaluator_validation,
                                               evaluator_test=evaluator_test)

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS=constructor_positional_args,
                CONSTRUCTOR_KEYWORD_ARGS={},
                FIT_POSITIONAL_ARGS=fixed_positional_args,
                FIT_KEYWORD_ARGS=fixed_keyword_args,
            )

            if URM_train_last_test is not None:
                recommender_input_args_last_test = recommender_input_args.copy()
                recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train_last_test
            else:
                recommender_input_args_last_test = None

            parameterSearch.search(
                recommender_input_args,
                recommender_input_args_last_test=recommender_input_args_last_test,
                fit_hyperparameters_values={},
                output_folder_path=output_folder_path,
                metric_to_optimize=metric_to_optimize,
                save_model=save_model,
                output_file_name_root=output_file_name_root
            )

            return

        ##########################################################################################################

        if is_crossvalidation:
            parameterSearch = SearchBayesianSkopt(Recommender_k_Fold_Wrapper,
                                                  is_early_stopping=is_earlystopping,
                                                  evaluator_validation=evaluator_validation,
                                                  evaluator_test=evaluator_test)
        else:
            parameterSearch = SearchBayesianSkopt(recommender_class,
                                                  is_early_stopping=is_earlystopping,
                                                  evaluator_validation=evaluator_validation,
                                                  evaluator_test=evaluator_test)

        hyperparameters_range_dictionary = None

        if recommender_class in [ItemKNNCF, UserKNNCF, ItemKNNCBF, ItemKNNCFCBFHybrid]:

            if "similarity" not in fixed_keyword_args.keys():
                fixed_keyword_args["similarity"] = "cosine"
                print("Parameter search: No similarity value given for {}, default set to cosine"
                      .format(recommender_class.RECOMMENDER_NAME))

            hyperparameters_range_dictionary = _knn_similarity_helper(fixed_keyword_args["similarity"])

            if recommender_class is ItemKNNCFCBFHybrid:
                hyperparameters_range_dictionary["ICM_weight"] = Real(low=1e-2, high=1e2, prior='log-uniform')

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS=constructor_positional_args,
                CONSTRUCTOR_KEYWORD_ARGS={},
                FIT_POSITIONAL_ARGS=fixed_positional_args,
                FIT_KEYWORD_ARGS=fixed_keyword_args,
            )

        ##########################################################################################################

        if recommender_class is P3alpha:
            hyperparameters_range_dictionary = {
                "topK": Integer(50, 500, prior='log-uniform'),
                "alpha": Real(low=0.1, high=2, prior='uniform'),
                "normalize_similarity": Categorical([True, False]),
            }

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS=constructor_positional_args,
                CONSTRUCTOR_KEYWORD_ARGS={},
                FIT_POSITIONAL_ARGS=fixed_positional_args,
                FIT_KEYWORD_ARGS=fixed_keyword_args
            )

        ##########################################################################################################

        if recommender_class is RP3beta:
            hyperparameters_range_dictionary = {
                "topK": Integer(50, 500, prior='log-uniform'),
                "alpha": Real(low=0.1, high=2, prior='uniform'),
                "beta": Real(low=0.1, high=1.5, prior='uniform'),
                "normalize_similarity": Categorical([True]),
            }

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS=constructor_positional_args,
                CONSTRUCTOR_KEYWORD_ARGS={},
                FIT_POSITIONAL_ARGS=fixed_positional_args,
                FIT_KEYWORD_ARGS=fixed_keyword_args,
            )

        ##########################################################################################################

        if recommender_class is EASE_R:
            hyperparameters_range_dictionary = {}
            hyperparameters_range_dictionary["topK"] = Integer(50, 1000, prior='log-uniform')
            hyperparameters_range_dictionary["normalize_matrix"] = Categorical([False])
            hyperparameters_range_dictionary["l2_norm"] = Real(low=1e0, high=1e7, prior='log-uniform')

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS=constructor_positional_args,
                CONSTRUCTOR_KEYWORD_ARGS={},
                FIT_POSITIONAL_ARGS=fixed_positional_args,
                FIT_KEYWORD_ARGS=fixed_keyword_args,
            )

        ##########################################################################################################

        if recommender_class is FunkSVD:
            hyperparameters_range_dictionary = {
                "sgd_mode": Categorical(["sgd", "adagrad", "adam"]),
                "epochs": Categorical([500]),
                "use_bias": Categorical([True, False]),
                "batch_size": Integer(1, 1024, prior='log-uniform'),
                "num_factors": Integer(8, 512, prior='log-uniform'),
                "item_reg": Real(low=1e-5, high=1e-2, prior='log-uniform'),
                "user_reg": Real(low=1e-5, high=1e-2, prior='log-uniform'),
                "learning_rate": Real(low=1e-4, high=1e-1, prior='log-uniform'),
                "negative_interactions_quota": Real(low=0.0, high=0.5, prior='uniform'),
                "dropout_quota": Real(low=0.0, high=0.7, prior='uniform'),
            }

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS=constructor_positional_args,
                CONSTRUCTOR_KEYWORD_ARGS={},
                FIT_POSITIONAL_ARGS=fixed_positional_args,
                FIT_KEYWORD_ARGS=fixed_keyword_args
            )

        ##########################################################################################################

        if recommender_class is AsySVD:
            hyperparameters_range_dictionary = {
                "sgd_mode": Categorical(["sgd", "adagrad", "adam"]),
                "epochs": Categorical([500]),
                "use_bias": Categorical([True, False]),
                "batch_size": Categorical([1]),
                "num_factors": Integer(8, 512, prior='log-uniform'),
                "item_reg": Real(low=1e-5, high=1e-2, prior='log-uniform'),
                "user_reg": Real(low=1e-5, high=1e-2, prior='log-uniform'),
                "learning_rate": Real(low=1e-4, high=1e-1, prior='log-uniform'),
                "negative_interactions_quota": Real(low=0.0, high=0.5, prior='uniform'),
            }

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS=constructor_positional_args,
                CONSTRUCTOR_KEYWORD_ARGS={},
                FIT_POSITIONAL_ARGS=fixed_positional_args,
                FIT_KEYWORD_ARGS=fixed_keyword_args
            )

        ##########################################################################################################

        if recommender_class is BPRMF:
            fixed_keyword_args["positive_threshold_BPR"] = None

            hyperparameters_range_dictionary = {
                "sgd_mode": Categorical(["sgd", "adagrad", "adam"]),
                "epochs": Categorical([500]),
                "num_factors": Integer(8, 512, prior='log-uniform'),
                "batch_size": Integer(1, 1024, prior='log-uniform'),
                "positive_reg": Real(low=1e-5, high=1e-2, prior='log-uniform'),
                "negative_reg": Real(low=1e-5, high=1e-2, prior='log-uniform'),
                "learning_rate": Real(low=1e-4, high=1e-1, prior='log-uniform'),
                "dropout_quota": Real(low=0.0, high=0.7, prior='uniform'),
            }

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS=constructor_positional_args,
                CONSTRUCTOR_KEYWORD_ARGS={},
                FIT_POSITIONAL_ARGS=fixed_positional_args,
                FIT_KEYWORD_ARGS=fixed_keyword_args
            )

        if recommender_class is BPRMF_AFM:

            hyperparameters_range_dictionary = {}
            hyperparameters_range_dictionary["num_factors"] = Integer(8, 512, prior="log-uniform")
            hyperparameters_range_dictionary["learning_rate"] = Real(low=1e-5, high=1e-2, prior="log-uniform")
            hyperparameters_range_dictionary["sgd_mode"] = Categorical(["sgd"])
            hyperparameters_range_dictionary["user_reg"] = Real(low=1e-8, high=1e-2, prior="log-uniform")
            hyperparameters_range_dictionary["feature_reg"] = Real(low=1e-8, high=1e-2, prior="log-uniform")
            hyperparameters_range_dictionary["epochs"] = Categorical([500])

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS=constructor_positional_args,
                CONSTRUCTOR_KEYWORD_ARGS={},
                FIT_POSITIONAL_ARGS=fixed_positional_args,
                FIT_KEYWORD_ARGS=fixed_keyword_args
            )

        ##########################################################################################################

        if recommender_class is IALS:
            hyperparameters_range_dictionary = {
                "num_factors": Integer(8, 512, prior='log-uniform'),
                "confidence_scaling": Categorical(["linear", "log"]),
                "alpha": Real(low=1e-3, high=50.0, prior='log-uniform'),
                "epsilon": Real(low=1e-3, high=10.0, prior='log-uniform'),
                "reg": Real(low=1e-5, high=1e-2, prior='log-uniform'),
            }

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS=constructor_positional_args,
                CONSTRUCTOR_KEYWORD_ARGS={},
                FIT_POSITIONAL_ARGS=fixed_positional_args,
                FIT_KEYWORD_ARGS=fixed_keyword_args
            )

        ##########################################################################################################

        if recommender_class is PureSVD:
            hyperparameters_range_dictionary = {
                "num_factors": Integer(8, 512, prior='log-uniform')
            }

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS=constructor_positional_args,
                CONSTRUCTOR_KEYWORD_ARGS={},
                FIT_POSITIONAL_ARGS=fixed_positional_args,
                FIT_KEYWORD_ARGS=fixed_keyword_args,
            )

        if recommender_class is PureSVDSimilarity:
            hyperparameters_range_dictionary = {
                "topK": Integer(50, 500, prior='log-uniform'),
                "num_factors": Integer(8, 512, prior='log-uniform')
            }

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS=constructor_positional_args,
                CONSTRUCTOR_KEYWORD_ARGS={},
                FIT_POSITIONAL_ARGS=fixed_positional_args,
                FIT_KEYWORD_ARGS=fixed_keyword_args,
            )

        ##########################################################################################################

        if recommender_class is NMF:
            hyperparameters_range_dictionary = {
                "num_factors": Integer(8, 512, prior='log-uniform'),
                "solver": Categorical(["coordinate_descent", "multiplicative_update"]),
                "init_type": Categorical(["random", "nndsvda"]),
                "beta_loss": Categorical(["frobenius", "kullback-leibler"]),
            }

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS=constructor_positional_args,
                CONSTRUCTOR_KEYWORD_ARGS={},
                FIT_POSITIONAL_ARGS=fixed_positional_args,
                FIT_KEYWORD_ARGS=fixed_keyword_args,
            )

        #########################################################################################################

        if recommender_class is SLIM_BPR:
            fixed_keyword_args["positive_threshold_BPR"] = None
            fixed_keyword_args["train_with_sparse_weights"] = None

            hyperparameters_range_dictionary = {
                "topK": Integer(50, 500, prior='log-uniform'),
                "epochs": Categorical([500]),
                "symmetric": Categorical([True, False]),
                "sgd_mode": Categorical(["sgd", "adagrad", "adam"]),
                "lambda_i": Real(low=1e-5, high=1e-2, prior='log-uniform'),
                "lambda_j": Real(low=1e-5, high=1e-2, prior='log-uniform'),
                "learning_rate": Real(low=1e-4, high=1e-1, prior='log-uniform'),
            }

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS=constructor_positional_args,
                CONSTRUCTOR_KEYWORD_ARGS={},
                FIT_POSITIONAL_ARGS=fixed_positional_args,
                FIT_KEYWORD_ARGS=fixed_keyword_args
            )

        ##########################################################################################################

        if recommender_class is SLIM_ElasticNet:
            hyperparameters_range_dictionary = {
                "topK": Integer(50, 500, prior='log-uniform'),
                "l1_ratio": Real(low=1e-5, high=1.0, prior='log-uniform'),
                "alpha": Real(low=1e-3, high=10.0, prior='uniform'),
            }

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS=constructor_positional_args,
                CONSTRUCTOR_KEYWORD_ARGS={},
                FIT_POSITIONAL_ARGS=fixed_positional_args,
                FIT_KEYWORD_ARGS=fixed_keyword_args,
            )

        if recommender_class in [FactorizationMachine, FactorizationMachineProfile]:

            hyperparameters_range_dictionary = {}
            hyperparameters_range_dictionary["rank"] = Integer(4, 200, prior='log-uniform')
            hyperparameters_range_dictionary["add_zeros_quota"] = Real(low=0., high=4., prior="uniform")
            hyperparameters_range_dictionary["learning_rate"] = Real(low=1e-6, high=1e-2, prior="log-uniform")
            hyperparameters_range_dictionary["reg"] = Real(low=1e-8, high=1e-2, prior="log-uniform")
            hyperparameters_range_dictionary["problem_type"] = Categorical(["classification", "regression"])
            hyperparameters_range_dictionary["normalize_instance"] = Categorical([True, False])
            hyperparameters_range_dictionary["epochs"] = Categorical([200])

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS=constructor_positional_args,
                CONSTRUCTOR_KEYWORD_ARGS={},
                FIT_POSITIONAL_ARGS=fixed_positional_args,
                FIT_KEYWORD_ARGS=fixed_keyword_args
            )

        if recommender_class in [NeuralFeatureCombiner, NeuralFeatureCombinerProfile, NeuralFeatureCombinerProfileCE,
                                 NeuralFeatureCombinerProfileBPR, NeuralFeatureCombinerProfileBPRMax]:

            hyperparameters_range_dictionary = {}
            hyperparameters_range_dictionary["topK"] = Integer(50, 500, prior='log-uniform')
            hyperparameters_range_dictionary["learning_rate"] = Real(low=1e-6, high=1e-3, prior="log-uniform")
            hyperparameters_range_dictionary["l2_reg"] = Real(low=1e-6, high=1e-2, prior="log-uniform")
            hyperparameters_range_dictionary["dropout"] = Real(low=0., high=0.8, prior="uniform")
            if recommender_class in [NeuralFeatureCombinerProfile, NeuralFeatureCombinerProfileCE,
                                     NeuralFeatureCombinerProfileBPR, NeuralFeatureCombinerProfileBPRMax]:
                hyperparameters_range_dictionary["batch_size"] = Integer(128, 1024, prior='log-uniform')
            else:
                hyperparameters_range_dictionary["batch_size"] = Categorical([24])
                hyperparameters_range_dictionary["add_zeros_quota"] = Real(low=1e-3, high=0.5, prior="log-uniform")
            hyperparameters_range_dictionary["epochs"] = Categorical([200])
            for i in range(kwargs['encoder_layers'] - 1):
                hyperparameters_range_dictionary["encoder_layer_{}".format(i)] = Integer(128, 1024, prior='log-uniform')
            hyperparameters_range_dictionary["encoder_layer_{}".format(kwargs['encoder_layers'] - 1)] = \
                Integer(64, 512, prior='log-uniform')
            for i in range(kwargs['decoder_layers']):
                hyperparameters_range_dictionary["decoder_layer_{}".format(i)] = Integer(4, 32, prior='log-uniform')

            if recommender_class in [NeuralFeatureCombinerProfile, NeuralFeatureCombinerProfileCE]:
                hyperparameters_range_dictionary["add_zeros_quota"] = Real(low=0., high=4., prior="uniform")

            if recommender_class in [NeuralFeatureCombinerProfileBPR, NeuralFeatureCombinerProfileBPRMax]:
                hyperparameters_range_dictionary["bpr_pop_alpha"] = Real(low=0., high=1., prior="uniform")

            if recommender_class in [NeuralFeatureCombinerProfileBPRMax]:
                hyperparameters_range_dictionary["neg_samples"] = Integer(2, 16, prior="log-uniform")

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS=constructor_positional_args,
                CONSTRUCTOR_KEYWORD_ARGS={},
                FIT_POSITIONAL_ARGS=fixed_positional_args,
                FIT_KEYWORD_ARGS=fixed_keyword_args
            )

        if recommender_class in [NeuralFeatureCombinerLight, NeuralFeatureCombinerLightProfile]:

            hyperparameters_range_dictionary = {}
            hyperparameters_range_dictionary["topK"] = Integer(50, 500, prior='log-uniform')
            hyperparameters_range_dictionary["learning_rate"] = Real(low=1e-6, high=1e-3, prior="log-uniform")
            hyperparameters_range_dictionary["add_zeros_quota"] = Real(low=0., high=4., prior="uniform")
            hyperparameters_range_dictionary["l2_reg"] = Real(low=1e-6, high=1e-2, prior="log-uniform")
            hyperparameters_range_dictionary["dropout"] = Real(low=0., high=0.8, prior="uniform")
            hyperparameters_range_dictionary["batch_size"] = Integer(128, 1024, prior='log-uniform')
            hyperparameters_range_dictionary["epochs"] = Categorical([200])
            for i in range(kwargs['layers']):
                hyperparameters_range_dictionary["layer_{}".format(i)] = Integer(64, 1024, prior='log-uniform')

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS=constructor_positional_args,
                CONSTRUCTOR_KEYWORD_ARGS={},
                FIT_POSITIONAL_ARGS=fixed_positional_args,
                FIT_KEYWORD_ARGS=fixed_keyword_args
            )

        if recommender_class in [WideAndDeep, WideAndDeepProfile]:

            hyperparameters_range_dictionary = {}
            hyperparameters_range_dictionary["user_embedding_size"] = Integer(32, 800, prior='log-uniform')
            hyperparameters_range_dictionary["item_embedding_size"] = Integer(32, 800, prior='log-uniform')
            hyperparameters_range_dictionary["batch_size"] = Integer(128, 1024, prior='log-uniform')
            hyperparameters_range_dictionary["learning_rate"] = Real(low=1e-6, high=1e-2, prior="log-uniform")
            hyperparameters_range_dictionary["add_zeros_quota"] = Real(low=0., high=4., prior="uniform")
            hyperparameters_range_dictionary["dropout"] = Real(low=0.1, high=0.8, prior="uniform")
            hyperparameters_range_dictionary["epochs"] = Categorical([200])
            for i in range(kwargs['layers']):
                hyperparameters_range_dictionary["layers_{}".format(i)] = Integer(64, 1024, prior="log-uniform")

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS=constructor_positional_args,
                CONSTRUCTOR_KEYWORD_ARGS={},
                FIT_POSITIONAL_ARGS=fixed_positional_args,
                FIT_KEYWORD_ARGS=fixed_keyword_args
            )

        if recommender_class is HP3:
            hyperparameters_range_dictionary = {}
            hyperparameters_range_dictionary["topK"] = Integer(50, 500, prior="log-uniform")
            hyperparameters_range_dictionary["add_zeros_quota"] = Real(low=0., high=1., prior="uniform")
            hyperparameters_range_dictionary["learning_rate"] = Real(low=1e-10, high=1e-2, prior="log-uniform")
            hyperparameters_range_dictionary["sgd_mode"] = Categorical(["sgd"])
            hyperparameters_range_dictionary["l1_reg"] = Real(low=1e-8, high=1e-2, prior="log-uniform")
            hyperparameters_range_dictionary["l2_reg"] = Real(low=1e-8, high=1e-2, prior="log-uniform")
            hyperparameters_range_dictionary["epochs"] = Categorical([200])

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS=constructor_positional_args,
                CONSTRUCTOR_KEYWORD_ARGS={},
                FIT_POSITIONAL_ARGS=fixed_positional_args,
                FIT_KEYWORD_ARGS=fixed_keyword_args
            )

        if recommender_class is FBSM:
            hyperparameters_range_dictionary = {}
            hyperparameters_range_dictionary["topK"] = Integer(50, 500, prior="log-uniform")
            hyperparameters_range_dictionary["n_factors"] = Integer(1, 10)
            hyperparameters_range_dictionary["init_type"] = Categorical(["random"])

            hyperparameters_range_dictionary["learning_rate"] = Real(low=1e-6, high=1e-2, prior="log-uniform")
            hyperparameters_range_dictionary["sgd_mode"] = Categorical(["sgd"])
            hyperparameters_range_dictionary["l2_reg_D"] = Real(low=1e-8, high=1e-2, prior="log-uniform")
            hyperparameters_range_dictionary["l2_reg_V"] = Real(low=1e-8, high=1e-2, prior="log-uniform")
            hyperparameters_range_dictionary["epochs"] = Categorical([200])

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS=constructor_positional_args,
                CONSTRUCTOR_KEYWORD_ARGS={},
                FIT_POSITIONAL_ARGS=fixed_positional_args,
                FIT_KEYWORD_ARGS=fixed_keyword_args
            )

        if recommender_class is LCE:
            hyperparameters_range_dictionary = {}
            hyperparameters_range_dictionary["k"] = Integer(10, 1000, prior="log-uniform")

            hyperparameters_range_dictionary["alpha"] = Real(low=0, high=1, prior="uniform")
            hyperparameters_range_dictionary["beta"] = Real(low=1e-5, high=0.5, prior="log-uniform")
            hyperparameters_range_dictionary["lamb"] = Real(low=1e-5, high=0.5, prior="log-uniform")
            hyperparameters_range_dictionary["epsilon"] = Real(low=1e-5, high=0.1, prior="log-uniform")
            hyperparameters_range_dictionary["maxiter"] = Integer(100, 1000, prior="log-uniform")

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS=constructor_positional_args,
                CONSTRUCTOR_KEYWORD_ARGS={},
                FIT_POSITIONAL_ARGS=fixed_positional_args,
                FIT_KEYWORD_ARGS=fixed_keyword_args
            )

        if recommender_class is DCT:
            hyperparameters_range_dictionary = {}
            hyperparameters_range_dictionary["num_factors"] = Integer(5, 1000, prior="log-uniform")
            hyperparameters_range_dictionary["num_eigs"] = Integer(5, 1000, prior="log-uniform")

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS=constructor_positional_args,
                CONSTRUCTOR_KEYWORD_ARGS={},
                FIT_POSITIONAL_ARGS=fixed_positional_args,
                FIT_KEYWORD_ARGS=fixed_keyword_args
            )

        if recommender_class is CFW_D:
            hyperparameters_range_dictionary = {}
            hyperparameters_range_dictionary["topK"] = Integer(50, 500, prior="log-uniform")

            hyperparameters_range_dictionary["init_type"] = Categorical(["random", "TF-IDF", "BM25"])
            hyperparameters_range_dictionary["learning_rate"] = Real(low=1e-6, high=1e-2, prior="log-uniform")
            hyperparameters_range_dictionary["sgd_mode"] = Categorical(["sgd"])
            hyperparameters_range_dictionary["l1_reg"] = Real(low=1e-8, high=1e-2, prior="log-uniform")
            hyperparameters_range_dictionary["l2_reg"] = Real(low=1e-8, high=1e-2, prior="log-uniform")
            hyperparameters_range_dictionary["epochs"] = Categorical([200])

            hyperparameters_range_dictionary["init_type"] = Categorical(["one", "random"])
            hyperparameters_range_dictionary["add_zeros_quota"] = Real(low=0., high=1.0, prior="uniform")
            hyperparameters_range_dictionary["positive_only_weights"] = Categorical([True, False])
            hyperparameters_range_dictionary["normalize_similarity"] = Categorical([True])

            hyperparameters_range_dictionary["use_dropout"] = Categorical([True])
            hyperparameters_range_dictionary["dropout_perc"] = Real(low=0., high=0.6, prior="uniform")

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS=constructor_positional_args,
                CONSTRUCTOR_KEYWORD_ARGS={},
                FIT_POSITIONAL_ARGS=fixed_positional_args,
                FIT_KEYWORD_ARGS=fixed_keyword_args
            )

        if recommender_class is CFW_DVV:

            hyperparameters_range_dictionary = {}
            hyperparameters_range_dictionary["topK"] = Integer(50, 500, prior="log-uniform")
            hyperparameters_range_dictionary["n_factors"] = Integer(1, 10)

            hyperparameters_range_dictionary["init_type"] = Categorical(["random", "TF-IDF", "BM25"])
            hyperparameters_range_dictionary["learning_rate"] = Real(low=1e-5, high=1e-2, prior="log-uniform")
            hyperparameters_range_dictionary["sgd_mode"] = Categorical(["sgd"])
            hyperparameters_range_dictionary["l2_reg_D"] = Real(low=1e-8, high=1e-2, prior="log-uniform")
            hyperparameters_range_dictionary["l2_reg_V"] = Real(low=1e-8, high=1e-2, prior="log-uniform")
            hyperparameters_range_dictionary["epochs"] = Categorical([200])

            hyperparameters_range_dictionary["add_zeros_quota"] = Real(low=0., high=1.0, prior="uniform")

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS=constructor_positional_args,
                CONSTRUCTOR_KEYWORD_ARGS={},
                FIT_POSITIONAL_ARGS=fixed_positional_args,
                FIT_KEYWORD_ARGS=fixed_keyword_args
            )

        if hyperparameters_range_dictionary is None:
            raise Exception("Unknown parameters to optimize for recommender {}"
                            .format(recommender_class.RECOMMENDER_NAME))

        hyperparameters_range_dictionary = _build_ranges_dictionary(
            hyperparameters_range_dictionary,
            fixed_keyword_args
        )

        if issubclass(recommender_class, EarlyStoppingModel):
            # Adding early stopping arguments
            recommender_input_args.add_fit_keyword_args({
                "validation_every_n": 5,
                "stop_on_validation": True,
                "evaluator_object": evaluator_validation_earlystopping,
                "lower_validations_allowed": 2,
                "validation_metric": metric_to_optimize,
            })

        if URM_train_last_test is not None:
            recommender_input_args_last_test = recommender_input_args.copy()
            recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train_last_test
        else:
            recommender_input_args_last_test = None

        ## Final step, after the hyperparameter range has been defined for each type of algorithm
        parameterSearch.search(recommender_input_args,
                               parameter_search_space=hyperparameters_range_dictionary,
                               n_cases=n_cases,
                               n_random_starts=n_random_starts,
                               output_folder_path=output_folder_path,
                               output_file_name_root=output_file_name_root,
                               resume_from_saved=resume_from_saved,
                               metric_to_optimize=metric_to_optimize,
                               save_model=save_model,
                               evaluate_on_test_each_best_solution=False,
                               recommender_input_args_last_test=recommender_input_args_last_test)

    except Exception as e:

        print("On recommender {} Exception {}".format(recommender_class, str(e)))
        traceback.print_exc()

        error_file = open(output_folder_path + "ErrorLog.txt", "a")
        error_file.write("On recommender {} Exception {}\n".format(recommender_class, str(e)))
        error_file.close()
