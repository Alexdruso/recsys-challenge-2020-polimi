
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 07/12/2020

@author: Alessandro Sanvito
"""

from ..KNN.ItemKNNSimilarityHybridRecommender import ItemKNNSimilarityHybridRecommender
from ..Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender

class GeneralizedSimilarityMergedHybridRecommender(BaseItemSimilarityMatrixRecommender):
    """
    Generalized similarity merged hybrud
    """

    RECOMMENDER_NAME = "GeneralizedSimilarityMergedHybridRecommender"

    def __init__(
            self,
            URM_train,
            similarityRecommenders : list,
            verbose=True
    ):
        self.RECOMMENDER_NAME = ''
        for recommender in similarityRecommenders:
            self.RECOMMENDER_NAME = self.RECOMMENDER_NAME + recommender.RECOMMENDER_NAME[:-11]
        self.RECOMMENDER_NAME = self.RECOMMENDER_NAME + 'HybridRecommender'

        super(GeneralizedSimilarityMergedHybridRecommender, self).__init__(
            URM_train,
            verbose=verbose
        )

        self.similarityRecommenders = similarityRecommenders

    def fit(self, alphas = None, topKs = None):

        recommender = ItemKNNSimilarityHybridRecommender(
            URM_train=self.URM_train,
            Similarity_1=self.similarityRecommenders[0].W_sparse,
            Similarity_2=self.similarityRecommenders[1].W_sparse,
            verbose=self.verbose
        )

        recommender.fit(
            topKs[0],
            alphas[0]
        )

        for index in range(1, len(alphas)):

            recommender = ItemKNNSimilarityHybridRecommender(
                URM_train=self.URM_train,
                Similarity_1=recommender.W_sparse,
                Similarity_2=self.similarityRecommenders[index+1].W_sparse,
                verbose=self.verbose
            )

            recommender.fit(
                topKs[index],
                alphas[index]
            )

        self.W_sparse = recommender.W_sparse
