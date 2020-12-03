#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 21/11/2020

@author: Alessandro Sanvito
"""

from ..Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from .RP3betaRecommender import RP3betaRecommender

class RP3betaCBFRecommender(BaseItemSimilarityMatrixRecommender):
    """ Recommender based on random walks"""

    RECOMMENDER_NAME = "RP3betaCBFRecommender"

    def __init__(self, URM_train, ICM_train, verbose=True):
        super(RP3betaCBFRecommender, self).__init__(URM_train, verbose=verbose)

        self.ICM_train = ICM_train

    def fit(self, topK=100, alpha=1., beta=0.6, min_rating=0, implicit=False, normalize_similarity=False):
        calculator = RP3betaRecommender(self.ICM_train.T, verbose=self.verbose)
        calculator.fit(topK=topK, alpha=alpha, beta=beta, min_rating=min_rating, implicit=implicit, normalize_similarity=normalize_similarity)
        self.W_sparse = calculator.W_sparse