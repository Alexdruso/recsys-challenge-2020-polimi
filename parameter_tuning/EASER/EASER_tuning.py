from src.Base.Evaluation.Evaluator import EvaluatorHoldout
from src.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from src.Utils.load_ICM import load_ICM
from src.Utils.load_URM import load_URM

URM_all = load_URM("../../in/data_train.csv")
ICM_all = load_ICM("../../in/data_ICM_title_abstract.csv")
from src.Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample

URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)

evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10], verbose=False)

tuning_params = {
    "topK": (10, 150),
    "l2_norm": (1e2, 1e5)
}

recommender = EASE_R_Recommender(URM_train=URM_train, sparse_threshold_quota=1.0)


def BO_func(topK, l2_norm):
    recommender.fit(topK=int(topK), l2_norm=l2_norm, verbose=False)
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
    init_points=15,
    n_iter=3,
)

hyperparameters = optimizer.max['params']

recommender = EASE_R_Recommender(URM_train=URM_all)
recommender.fit(topK=int(hyperparameters['topK']), l2_norm=hyperparameters['l2_norm'])

recommender.save_model(folder_path='../../models/', file_name=recommender.RECOMMENDER_NAME)

import json

with open("logs/" + recommender.RECOMMENDER_NAME + "_logs.json", 'w') as json_file:
    json.dump(optimizer.max, json_file)