from src.Base.Evaluation.Evaluator import EvaluatorHoldout
from src.MatrixFactorization.PureSVDRecommender import PureSVDItemRecommender
from src.Utils.load_ICM import load_ICM
from src.Utils.load_URM import load_URM

URM_all = load_URM("../../in/data_train.csv")
ICM_all = load_ICM("../../in/data_ICM_title_abstract.csv")
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

tuning_params = {
    "topK": (10, 500),
    "num_factors": (10, 500)
}

recommender = PureSVDItemRecommender(URM_train=URM_train, verbose=False)


def BO_func(topK, num_factors):
    recommender.fit(topK=int(topK), num_factors=int(num_factors))
    result_dict, _ = evaluator_validation.evaluateRecommender(recommender)

    return result_dict[10]["MAP"]


from bayes_opt import BayesianOptimization

optimizer = BayesianOptimization(
    f=BO_func,
    pbounds=tuning_params,
    verbose=5,
    random_state=5,
)
optimizer.maximize(
    init_points=30,
    n_iter=10,
)

print(optimizer.max)