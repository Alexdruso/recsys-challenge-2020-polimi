#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15/11/2020

@author: Alessandro Sanvito
"""

from ..Base.BaseRecommender import BaseRecommender
from ..MatrixFactorization.PureSVDRecommender import PureSVDItemRecommender
from ..KNN.ItemKNNCBFCFSimilarityHybridRecommender import ItemKNNCBFCFSimilarityHybridRecommender


class CBFCFPureSVDPipelinedRecommender(BaseRecommender):
    """ This recommender first builds a new user rating matrix using a CBFCF hybrid, then it uses the new matrix to fit
        a PureSVD recommender
    """

    RECOMMENDER_NAME = "CBFCF+PureSVDPipelinedRecommender"

    def __init__(
            self,
            URM_train,
            ICM_train,
            topK_knncf=50,
            shrink_knncf=100,
            topK_knncbf=50,
            shrink_knncbf=100,
            similarity='cosine',
            topK=100,
            alpha=0.5,
            verbose=True
    ):
        super().__init__(URM_train)

        recommender = ItemKNNCBFCFSimilarityHybridRecommender(
            URM_train,
            ICM_train,
            topK_knncf=topK_knncf,
            shrink_knncf=shrink_knncf,
            topK_knncbf=topK_knncbf,
            shrink_knncbf=shrink_knncbf,
            similarity=similarity,
            verbose=verbose
        )

        recommender.fit(topK, alpha)

        self.recommender = PureSVDItemRecommender(recommender.URM_train.dot(recommender.W_sparse), verbose=verbose)

    def fit(
            self,
            num_factors=100,
            topK=None
    ):
        self.recommender.fit(num_factors=num_factors, topK=topK)
        self.recommender.URM_train = self.URM_train

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        self.recommender._compute_item_score(user_id_array=user_id_array, items_to_compute=items_to_compute)

    def save_model(self, folder_path, file_name=None):
        self.recommender.save_model(folder_path=folder_path,file_name=file_name)
