
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
            p3alpha_recommender: P3alphaRecommender,
            item_knncbf_recommender: ItemKNNCBFRecommender,
            verbose=True
    ):
        super(P3alphaCBFMergedHybridRecommender, self).__init__(
            URM_train,
            Similarity_1=p3alpha_recommender.W_sparse,
            Similarity_2=item_knncbf_recommender.W_sparse,
            verbose=verbose
        )
