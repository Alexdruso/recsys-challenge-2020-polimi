# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 17/11/2020

@author: Alessandro Sanvito
"""

from ..Base.BaseRecommender import BaseRecommender
from ..Base.NonPersonalizedRecommender import TopPop


class ColdUsersTopPop(BaseRecommender):
    """
    This recommender fuses a CBF approach and a P3alpha recommender
    """

    RECOMMENDER_NAME = "SimilarityMergedHybridRecommender"

    def __init__(
            self,
            URM_train,
            warm_recommender: BaseRecommender,
            verbose=True
    ):
        self.RECOMMENDER_NAME = warm_recommender.RECOMMENDER_NAME[:-11]+'ColdUsersTopPopRecommender'

        self.warm_recommender = warm_recommender

        self.cold_recommender = TopPop(URM_train=URM_train)

        super(ColdUsersTopPop, self).__init__(URM_train=URM_train, verbose=verbose)

        self.cold_users = self._get_cold_user_mask()

    def fit(self):
        self.cold_recommender.fit()

    def save_model(self, folder_path, file_name=None):
        pass

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        if self.cold_users[user_id_array]:
            return self.cold_recommender._compute_item_score(user_id_array, items_to_compute)
        else:
            return self.warm_recommender._compute_item_score(user_id_array, items_to_compute)