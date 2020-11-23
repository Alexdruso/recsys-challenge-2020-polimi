from src.Base.Evaluation.Evaluator import EvaluatorHoldout
from src.FeatureWeighting.CFW_D_Similarity_Linalg import CFW_D_Similarity_Linalg
from src.GraphBased.P3alphaRecommender import P3alphaRecommender
from src.Utils.load_ICM import load_ICM
from src.Utils.load_URM import load_URM
import json
import os

URM_all = load_URM("../../in/data_train.csv")
ICM_all = load_ICM("../../in/data_ICM_title_abstract.csv")
from src.Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample

URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)

evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10], verbose=False)

service_recommender = P3alphaRecommender(URM_train=URM_train, verbose=False)

with open('logs/'+service_recommender.RECOMMENDER_NAME+'_logs.json', 'r') as config_file:
    max = json.load(config_file)['params']

max['topK'] = int(max['topK'])

service_recommender.fit(**max, implicit=True)

print(max)

recommender = CFW_D_Similarity_Linalg(URM_train=URM_train, ICM=ICM_all, S_matrix_target=service_recommender.W_sparse)
tuning_params = {
    "topK": (10, 500)
}

def BO_func(topK):
    recommender.fit(topK=int(topK))
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
    init_points=3,
    n_iter=15,
)

hyperparameters = optimizer.max['params']

with open("logs/" + recommender.RECOMMENDER_NAME + "_logs.json", 'r') as json_file:
    data = json.load(json_file)
with open("logs/" + recommender.RECOMMENDER_NAME + "_logs.json", 'w') as json_file:
    if data['target'] < optimizer.max['target']:
        json.dump(optimizer.max, json_file)


#recommender = P3alphaRecommender(URM_train=URM_all, verbose=False)
#recommender.fit(topK=int(hyperparameters['topK']), alpha=hyperparameters['alpha'], implicit=True)

#recommender.save_model(folder_path='../models/', file_name=recommender.RECOMMENDER_NAME)
