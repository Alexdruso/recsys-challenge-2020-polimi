#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/11/2020

@author: Alessandro Sanvito
"""
from .Evaluator import Evaluator
from .Evaluator import EvaluatorHoldout

class K_Fold_Evaluator_MAP(Evaluator):

    def __init__(self, URM_test_list: list, cutoff_list, min_ratings_per_user=1, exclude_seen=True,
                 diversity_object=None, ignore_items=None, ignore_users_list = None, verbose=True):

        self.evaluator_list = []

        if ignore_users_list == None:
            ignore_users_list = [None]*len(URM_test_list)

        for index in range(len(URM_test_list)):

            self.evaluator_list.append(
                EvaluatorHoldout(
                    URM_test_list=URM_test_list[index],
                    cutoff_list=cutoff_list,
                    min_ratings_per_user=min_ratings_per_user,
                    exclude_seen=exclude_seen,
                    diversity_object=diversity_object,
                    ignore_items=ignore_items,
                    ignore_users=ignore_users_list[index],
                    verbose=verbose
                )
            )

    def evaluateRecommender(self, recommender_list : list):

        results = []

        for index in range(len(recommender_list)):
            result_dict, _ = self.evaluator_list[index].evaluateRecommender(
                recommender_list[index]
            )

            results.append(
                result_dict[10]["MAP"]
            )

        return results