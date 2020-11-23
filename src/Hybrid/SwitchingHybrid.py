# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 17/11/2020

@author: Alessandro Sanvito
"""

from ..Base.BaseRecommender import BaseRecommender
from ..Base.NonPersonalizedRecommender import TopPop


class SwitchingHybrid(BaseRecommender):
    """
    This recommender bundles different recommenders and switches to act on different user groups
    """

    RECOMMENDER_NAME = "SwitchingHybridRecommender"

    def __init__(
            self,
            URM_train,
            recommenders: list,
            users_categories: list,
    ):
        super(SwitchingHybrid, self).__init__(URM_train=URM_train, verbose=False)

        self.recommenders = recommenders

        self.RECOMMENDER_NAME = ""
        for recommender in recommenders:
            self.RECOMMENDER_NAME = self.RECOMMENDER_NAME + recommender.RECOMMENDER_NAME[:-11]

        self.RECOMMENDER_NAME = self.RECOMMENDER_NAME + "SwitchingHybridRecommender"

        self.user_categories = users_categories

    def fit(self):
        pass

    def save_model(self, folder_path, file_name=None):
        pass

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        import numpy as np
        scores_batch = []

        for user_id in user_id_array:
            for index in range(len(self.recommenders)):
                if user_id in self.user_categories[index]:
                    scores_batch.insert(
                        user_id,
                        self.recommenders[index]._compute_item_score(
                            user_id_array=user_id,
                            items_to_compute=items_to_compute
                        )
                    )

        return np.asarray(scores_batch)
