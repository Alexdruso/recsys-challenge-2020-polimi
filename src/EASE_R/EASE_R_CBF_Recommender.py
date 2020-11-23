#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/2020

@author: Alessandro Sanvito
"""

from ..Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from .EASE_R_Recommender import EASE_R_Recommender
class EASE_R_CBF_Recommender(BaseItemSimilarityMatrixRecommender):
    """ EASER CBF version"""

    RECOMMENDER_NAME = "EASERCBFRecommender"

    def __init__(self, URM_train, ICM_train, sparse_threshold_quota=1.0 , verbose=True):
        super(EASE_R_CBF_Recommender, self).__init__(URM_train, verbose=verbose)
        self.sparse_threshold_quota = sparse_threshold_quota
        self.ICM_train = ICM_train

    def fit(self, topK=100, l2_norm=1e3 , normalize_similarity=False):
        calculator = EASE_R_Recommender(self.ICM_train.T, sparse_threshold_quota= self.sparse_threshold_quota)
        calculator.fit(topK=topK, l2_norm=l2_norm, normalize_matrix=normalize_similarity, verbose=self.verbose)
        self.W_sparse = calculator.W_sparse