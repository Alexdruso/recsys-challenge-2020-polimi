
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15/11/2020

@author: Alessandro Sanvito
"""

from ..KNN.ItemKNNSimilarityHybridRecommender import ItemKNNSimilarityHybridRecommender
from ..GraphBased.P3alphaRecommender import P3alphaRecommender
from ..KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender


class P3alphaCBFMergedHybridRecommender(ItemKNNSimilarityHybridRecommender):
    """
    This recommender fuses a CBF approach and a P3alpha recommender
    """

    RECOMMENDER_NAME = "P3alphaCBFMergedHybridRecommender"

    def __init__(
            self,
            URM_train,
            ICM_train,
            topK_P3alpha = 100,
            alpha_P3alpha = 1.0,
            topK_knncbf = 50,
            shrink_knncbf = 100,
            similarity_knncbf='cosine',
            implicit=True,
            verbose=True
    ):

        P3alpha_recommender = P3alphaRecommender(URM_train=URM_train, verbose=verbose)

        P3alpha_recommender.fit(topK=topK_P3alpha,alpha=alpha_P3alpha,implicit=implicit)

        item_knncbf_recommender = ItemKNNCBFRecommender(URM_train=URM_train, ICM_train=ICM_train, verbose=verbose)

        item_knncbf_recommender.fit(topK=topK_knncbf,shrink=shrink_knncbf,similarity=similarity_knncbf)

        super(P3alphaCBFMergedHybridRecommender, self).__init__(
            URM_train,
            Similarity_1=P3alpha_recommender.W_sparse,
            Similarity_2=item_knncbf_recommender.W_sparse,
            verbose=verbose
        )
