from src.Base.Evaluation.Evaluator import EvaluatorHoldout
from src.Utils.load_ICM import load_ICM
from src.Utils.load_URM import load_URM
from src.Utils.ICM_preprocessing import *

URM_all = load_URM("../../../in/data_train.csv")
ICM_all = load_ICM("../../../in/data_ICM_title_abstract.csv")
from src.Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample

URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)

import numpy as np

profile_length = np.ediff1d(URM_train.indptr)
block_size = int(len(profile_length)*0.25)

start_pos = 3 * block_size
end_pos = min(4 * block_size, len(profile_length))
sorted_users = np.argsort(profile_length)

users_in_group = sorted_users[start_pos:end_pos]

users_in_group_p_len = profile_length[users_in_group]
sorted_users = np.argsort(profile_length)


users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
users_not_in_group = sorted_users[users_not_in_group_flag]

evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10], verbose=False, ignore_users = users_not_in_group)

from src.GraphBased.RP3betaCBFRecommender import RP3betaCBFRecommender

from bayes_opt import BayesianOptimization

binarize_ICM(ICM_all)

ICM_all = combine(ICM_all, URM_train)

rp3betaCBF_recommender = RP3betaCBFRecommender(URM_train=URM_train, ICM_train=ICM_all, verbose=False)

tuning_params = {
    "alpha": (0.1, 0.9),
    "beta": (0.1, 0.9),
    "topK": (10, 1000)
}


def BO_func(
        alpha,
        beta,
        topK
):
    rp3betaCBF_recommender.fit(alpha=alpha, beta=beta, topK=int(topK), implicit=True)
    result_dict, _ = evaluator_validation.evaluateRecommender(rp3betaCBF_recommender)

    return result_dict[10]["MAP"]


optimizer = BayesianOptimization(
    f=BO_func,
    pbounds=tuning_params,
    verbose=5,
    random_state=5,
)

optimizer.maximize(
    init_points=100,
    n_iter=40,
)

import json

with open("logs/FeatureCombined" + rp3betaCBF_recommender.RECOMMENDER_NAME + "_best25_logs.json", 'w') as json_file:
    json.dump(optimizer.max, json_file)
