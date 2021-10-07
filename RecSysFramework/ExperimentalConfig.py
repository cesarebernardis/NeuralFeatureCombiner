from RecSysFramework.DataManager.Reader import MovielensHetrecReader
from RecSysFramework.DataManager.Reader import BookCrossingReader
from RecSysFramework.DataManager.Reader import YahooMoviesReader, YahooMoviesReducedReader
from RecSysFramework.DataManager.Reader import AmazonGamesReader

from RecSysFramework.Recommender.KNN import ItemKNNCF
from RecSysFramework.Recommender.GraphBased import RP3beta
from RecSysFramework.Recommender.SLIM.ElasticNet import SLIM
from RecSysFramework.Recommender.KNN import EASE_R

from RecSysFramework.DataManager.Splitter import ColdItemsHoldout, Holdout, HybridItemHoldout
from RecSysFramework.DataManager.DatasetPostprocessing import CombinedKCore, ImplicitURM, UserSample

_DEFAULT_SPLIT = ColdItemsHoldout(train_perc=0.8, test_perc=0.2)
_DEFAULT_KCORE = CombinedKCore(user_k_core=5, item_urm_k_core=5, item_icm_k_core=5, feature_k_core=5, reshape=True)

EXPERIMENTAL_CONFIG = {
    'n_folds': 10,
    'cold_split': _DEFAULT_SPLIT,
    'cold_split_perc': [
        _DEFAULT_SPLIT,
        ColdItemsHoldout(train_perc=0.6, test_perc=0.4),
        ColdItemsHoldout(train_perc=0.4, test_perc=0.6),
        ColdItemsHoldout(train_perc=0.2, test_perc=0.8),
    ],
    'hybrid_split_perc': [
        HybridItemHoldout(train_perc=0.8, test_perc=0.2, cold_ratio=0.25),
        HybridItemHoldout(train_perc=0.8, test_perc=0.2, cold_ratio=0.5),
        HybridItemHoldout(train_perc=0.8, test_perc=0.2, cold_ratio=0.75),
        HybridItemHoldout(train_perc=0.8, test_perc=0.2, cold_ratio=0.),
    ],
    'warm_split': Holdout(train_perc=0.8, test_perc=0.2),
    'datasets': [
        {
        #    'datareader': YahooMoviesReader,
        #    'subsampling_postprocessings': None,
        #    'postprocessings': [
        #        ImplicitURM(min_rating_threshold=3.),
        #        _DEFAULT_KCORE,
        #    ]
        # }, {
        #     'datareader': MovielensHetrecReader,
        #     'subsampling_postprocessings': [
        #         UserSample(user_quota=0.5),
        #         _DEFAULT_KCORE,
        #     ],
        #     'postprocessings': [
        #         ImplicitURM(min_rating_threshold=3.),
        #         _DEFAULT_KCORE,
        #     ]
        # }, {
        #     'datareader': BookCrossingReader,
        #     'subsampling_postprocessings': [
        #         UserSample(user_quota=0.5),
        #         _DEFAULT_KCORE,
        #     ],
        #     'postprocessings': [
        #         ImplicitURM(min_rating_threshold=7.),
        #         _DEFAULT_KCORE,
        #     ]
        # }, {
             'datareader': AmazonGamesReader,
             'subsampling_postprocessings': [
                 UserSample(user_quota=0.5),
                 _DEFAULT_KCORE,
             ],
             'postprocessings': [
                 ImplicitURM(min_rating_threshold=3.),
                 _DEFAULT_KCORE,
             ]
        }
    ],
    'reduced_datasets': [
        {
            'datareader': YahooMoviesReducedReader,
            'subsampling_postprocessings': None,
            'postprocessings': [
                ImplicitURM(min_rating_threshold=3.),
                _DEFAULT_KCORE,
            ]
        }
    ],
    'collaborative_algorithms': [RP3beta, ItemKNNCF, SLIM, EASE_R],
    'recap_metrics': ["Precision", "Recall", "MAP", "NDCG", "Coverage Test Item"],
    'cutoffs': [5, 10, 25],
}
