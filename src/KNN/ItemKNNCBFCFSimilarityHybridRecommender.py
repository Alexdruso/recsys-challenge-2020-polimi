#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15/11/2020

@author: Alessandro Sanvito
"""

from .ItemKNNSimilarityHybridRecommender import ItemKNNSimilarityHybridRecommender
from .ItemKNNCFRecommender import ItemKNNCFRecommender
from .ItemKNNCBFRecommender import ItemKNNCBFRecommender


class ItemKNNCBFCFSimilarityHybridRecommender(ItemKNNSimilarityHybridRecommender):
    """
    This recommender fuses a CBF approach and an item based CF approach
    """

    RECOMMENDER_NAME = "ItemKNNCBFCFSimilarityHybridRecommender"

    def __init__(
            self,
            URM_train,
            ICM_train,
            topK_knncf = 50,
            shrink_knncf = 100,
            topK_knncbf = 50,
            shrink_knncbf = 100,
            similarity_knncbf='cosine',
            similarity_knncf='cosine',
            verbose=True
    ):

        item_knncf_recommender = ItemKNNCFRecommender(URM_train=URM_train, verbose=verbose)

        item_knncf_recommender.fit(topK=topK_knncf,shrink=shrink_knncf,similarity=similarity_knncf)

        item_knncbf_recommender = ItemKNNCBFRecommender(URM_train=URM_train, ICM_train=ICM_train, verbose=verbose)

        item_knncbf_recommender.fit(topK=topK_knncbf,shrink=shrink_knncbf,similarity=similarity_knncbf)

        super(ItemKNNCBFCFSimilarityHybridRecommender, self).__init__(
            URM_train,
            Similarity_1=item_knncf_recommender.W_sparse,
            Similarity_2=item_knncbf_recommender.W_sparse,
            verbose=verbose
        )
