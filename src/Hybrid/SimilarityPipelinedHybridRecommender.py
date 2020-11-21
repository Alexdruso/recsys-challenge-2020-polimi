#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19/11/2020

@author: Alessandro Sanvito
"""

from ..Base.Recommender_utils import check_matrix, similarityMatrixTopK
from ..Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender


class SimilarityPipelinedHybridRecommender(BaseItemSimilarityMatrixRecommender):
    """ Pipelined similarity recommender
    """

    RECOMMENDER_NAME = "SimilarityPipelinedHybridRecommender"

    def __init__(
            self,
            URM_train,
            recommender_1: BaseItemSimilarityMatrixRecommender,
            recommender_2: BaseItemSimilarityMatrixRecommender,
            verbose=True
    ):
        super(SimilarityPipelinedHybridRecommender, self).__init__(URM_train, verbose=verbose)

        self.RECOMMENDER_NAME = recommender_1.RECOMMENDER_NAME[:-11] + recommender_2.RECOMMENDER_NAME[
                                                                       :-11] + 'PipelinedHybridRecommender'

        Similarity_1 = recommender_1.W_sparse
        Similarity_2 = recommender_2.W_sparse

        if Similarity_1.shape != Similarity_2.shape:
            raise ValueError(
                "ItemKNNSimilarityHybridRecommender: similarities have different size, S1 is {}, S2 is {}".format(
                    Similarity_1.shape, Similarity_2.shape
                ))

        # CSR is faster during evaluation
        self.Similarity_1 = check_matrix(Similarity_1.copy(), 'csr')
        self.Similarity_2 = check_matrix(Similarity_2.copy(), 'csr')

    def fit(self, topK=100):
        self.topK = topK

        W_sparse = self.Similarity_1.dot(self.Similarity_2.T)

        self.W_sparse = similarityMatrixTopK(W_sparse, k=self.topK)
        self.W_sparse = check_matrix(self.W_sparse, format='csr')
