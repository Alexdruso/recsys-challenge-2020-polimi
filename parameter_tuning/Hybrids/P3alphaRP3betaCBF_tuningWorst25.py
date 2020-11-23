from scipy import sparse as sps

from src.Base.Evaluation.Evaluator import EvaluatorHoldout
from src.Utils.load_ICM import load_ICM
from src.Utils.load_URM import load_URM
from src.Utils.ICM_preprocessing import *

URM_all = load_URM("../../in/data_train.csv")
ICM_all = load_ICM("../../in/data_ICM_title_abstract.csv")
from src.Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample

URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)

import numpy as np

profile_length = np.ediff1d(URM_train.indptr)
block_size = int(len(profile_length)*0.25)

start_pos = 0 * block_size
end_pos = min(1 * block_size, len(profile_length))
sorted_users = np.argsort(profile_length)

users_in_group = sorted_users[start_pos:end_pos]

users_in_group_p_len = profile_length[users_in_group]
sorted_users = np.argsort(profile_length)


users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
users_not_in_group = sorted_users[users_not_in_group_flag]


evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10], verbose=False, ignore_users = users_not_in_group)



from src.Hybrid.SimilarityMergedHybridRecommender import SimilarityMergedHybridRecommender
from src.GraphBased.P3alphaRecommender import P3alphaRecommender
from src.GraphBased.RP3betaCBFRecommender import RP3betaCBFRecommender

from bayes_opt import BayesianOptimization

p3alpha_recommender = P3alphaRecommender(URM_train=URM_train, verbose=False)
p3alpha_recommender.fit(topK=228,alpha=0.512,implicit=True)

rp3betaCBF_recommender = RP3betaCBFRecommender(URM_train=URM_train, ICM_train=ICM_all, verbose=False)
rp3betaCBF_recommender.fit(topK=63,alpha=0.221,beta=0.341,implicit=False)

recommender_worst = SimilarityMergedHybridRecommender(URM_train=URM_train,CFRecommender=p3alpha_recommender,CBFRecommender=rp3betaCBF_recommender,verbose=False)

tuning_params = {
    "alpha": (0.1,0.9),
    "topKWorst25":(10,500)
}


def BO_func(
        alpha,
        topKWorst25,
):

    recommender_worst.fit(topK=int(topKWorst25), alpha=alpha)

    result_dict, _ = evaluator_validation.evaluateRecommender(recommender_worst)

    return result_dict[10]["MAP"]


optimizer = BayesianOptimization(
    f=BO_func,
    pbounds=tuning_params,
    verbose=5,
    random_state=5,
)

optimizer.maximize(
    init_points=150,
    n_iter=60,
)

import json

with open("logs/" + recommender_worst.RECOMMENDER_NAME + "_worst25_logs.json", 'w') as json_file:
    json.dump(optimizer.max, json_file)
