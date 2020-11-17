
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15/11/2020

@author: Alessandro Sanvito
"""

from ..KNN.ItemKNNSimilarityHybridRecommender import ItemKNNSimilarityHybridRecommender
from ..Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender

class SimilarityMergedHybridRecommender(ItemKNNSimilarityHybridRecommender):
    """
    This recommender fuses a CBF approach and a P3alpha recommender
    """

    RECOMMENDER_NAME = "SimilarityMergedHybridRecommender"

    def __init__(
            self,
            URM_train,
            CFRecommender: BaseItemSimilarityMatrixRecommender,
            CBFRecommender: BaseItemSimilarityMatrixRecommender,
            verbose=True
    ):

        self.RECOMMENDER_NAME = CFRecommender.RECOMMENDER_NAME[:-11]+CBFRecommender.RECOMMENDER_NAME[:-11]+'HybridRecommender'

        super(SimilarityMergedHybridRecommender, self).__init__(
            URM_train,
            Similarity_1=CFRecommender.W_sparse,
            Similarity_2=CBFRecommender.W_sparse,
            verbose=verbose
        )
