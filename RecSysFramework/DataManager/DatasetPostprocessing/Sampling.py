#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 30/11/18

@author: Cesare Bernardis
"""


import numpy as np

from .DatasetPostprocessing import DatasetPostprocessing


class UserSample(DatasetPostprocessing):

    """
    This class selects a partition of URM such that only some of the original users are present
    """


    def __init__(self, user_quota=1.0, reshape=True):

        assert user_quota > 0.0 and user_quota <= 1.0,\
            "DataReaderPostprocessing - User sample: user_quota must be a positive value > 0.0 and <= 1.0, " \
            "provided value was {}".format(user_quota)

        super(UserSample, self).__init__()
        self.user_quota = user_quota
        self.reshape = reshape


    def get_name(self):
        return "user_sample_{}{}".format(self.user_quota, "_reshaped" if self.reshape else "")


    def _get_balanced_subsample(self, urm, quota, rounds=10):

        n_users, n_items = urm.shape
        interactions = np.ediff1d(urm.tocsr().indptr)

        users_to_remove = []
        best_error = np.inf
        for _ in range(rounds):
            remove_quota = 1. - quota
            utr = np.random.choice(n_users, int(n_users * remove_quota), replace=False)
            error = np.square(interactions[utr].sum() - remove_quota * interactions.sum())
            if error < best_error:
                best_error = error
                users_to_remove = utr

        return users_to_remove


    def apply(self, dataset, random_seed=42):

        if random_seed is not None:
            np.random.seed(random_seed)

        print("DatasetPostprocessing - UserSample: Sampling {:.2f}% of users".format(self.user_quota * 100))
        users_to_remove = self._get_balanced_subsample(dataset.get_URM(), self.user_quota)

        new_dataset = dataset.copy()
        new_dataset.remove_users(users_to_remove, keep_original_shape=not self.reshape)
        new_dataset.add_postprocessing(self)

        return new_dataset



class ItemSample(DatasetPostprocessing):

    """
    This class selects a partition of URM such that only some of the original items are present
    """


    def __init__(self, item_quota=1.0, reshape=True):

        assert item_quota > 0.0 and item_quota <= 1.0,\
            "DataReaderPostprocessing - Item sample: item_quota must be a positive value > 0.0 and <= 1.0, " \
            "provided value was {}".format(item_quota)

        super(ItemSample, self).__init__()
        self.item_quota = item_quota
        self.reshape = reshape


    def get_name(self):
        return "item_sample_{}{}".format(self.item_quota, "_reshaped" if self.reshape else "")


    def _get_balanced_subsample(self, urm, quota, rounds=10):

        n_users, n_items = urm.shape
        interactions = np.ediff1d(urm.tocsc().indptr)

        items_to_remove = []
        best_error = np.inf
        for _ in range(rounds):
            remove_quota = 1. - quota
            itr = np.random.choice(n_items, int(n_items * remove_quota), replace=False)
            error = np.square(interactions[itr].sum() - remove_quota * interactions.sum())
            if error < best_error:
                best_error = error
                items_to_remove = itr

        return items_to_remove


    def apply(self, dataset, random_seed=42):

        if random_seed is not None:
            np.random.seed(random_seed)

        print("DatasetPostprocessing - ItemSample: Sampling {:.2f}% of items".format(self.item_quota * 100))
        items_to_remove = self._get_balanced_subsample(dataset.get_URM(), self.item_quota)

        new_dataset = dataset.copy()
        new_dataset.remove_items(items_to_remove, keep_original_shape=not self.reshape)
        new_dataset.add_postprocessing(self)

        return new_dataset
