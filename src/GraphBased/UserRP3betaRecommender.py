#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 01/12/2020

@author: Alessandro Sanvito
"""

from ..Base.BaseSimilarityMatrixRecommender import BaseUserSimilarityMatrixRecommender
from .RP3betaRecommender import RP3betaRecommender

class UserRP3betaRecommender(BaseUserSimilarityMatrixRecommender):
    """ Recommender based on random walks"""

    RECOMMENDER_NAME = "UserRP3betaRecommender"

    def __init__(self, URM_train, verbose=True):
        super(UserRP3betaRecommender, self).__init__(URM_train, verbose=verbose)


    def fit(self, topK=100, alpha=1., beta=0.6, min_rating=0, implicit=False, normalize_similarity=False):
        calculator = RP3betaRecommender(self.URM_train.T, verbose=self.verbose)
        calculator.fit(topK=topK, alpha=alpha, beta=beta, min_rating=min_rating, implicit=implicit, normalize_similarity=normalize_similarity)
        self.W_sparse = calculator.W_sparse