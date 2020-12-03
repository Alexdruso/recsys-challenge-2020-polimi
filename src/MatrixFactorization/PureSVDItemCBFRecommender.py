#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 30/11/2020

@author: Alessandro Sanvito
"""

from ..Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from .PureSVDRecommender import PureSVDItemRecommender
class PureSVDItemCBFRecommender(BaseItemSimilarityMatrixRecommender):
    """ PureSVDItem recommender"""

    RECOMMENDER_NAME = "PureSVDItemCBFRecommender"

    def __init__(self, URM_train, ICM_train, verbose=True):
        super(PureSVDItemCBFRecommender, self).__init__(URM_train, verbose=verbose)

        self.ICM_train = ICM_train

    def fit(self, num_factors=100, topK=None, random_seed=None):
        calculator = PureSVDItemRecommender(self.ICM_train.T, verbose=self.verbose)
        calculator.fit(num_factors=num_factors, topK=topK, random_seed=random_seed)
        self.W_sparse = calculator.W_sparse