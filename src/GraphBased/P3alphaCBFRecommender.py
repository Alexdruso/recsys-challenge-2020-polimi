#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 18/11/2020

@author: Alessandro Sanvito
"""

from ..Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from .P3alphaRecommender import P3alphaRecommender

class P3alphaCBFRecommender(BaseItemSimilarityMatrixRecommender):
    """ ItemKNN recommender"""

    RECOMMENDER_NAME = "P3alphaCBFRecommender"

    def __init__(self, URM_train, ICM_train, verbose=True):
        super(P3alphaCBFRecommender, self).__init__(URM_train, verbose=verbose)

        self.ICM_train = ICM_train

    def fit(self, topK=100, alpha=1., min_rating=0, implicit=False, normalize_similarity=False):
        calculator = P3alphaRecommender(self.ICM_train.T, verbose=self.verbose)
        calculator.fit(topK=topK, alpha=alpha, min_rating=min_rating, implicit=implicit, normalize_similarity=normalize_similarity)
        self.W_sparse = calculator.W_sparse