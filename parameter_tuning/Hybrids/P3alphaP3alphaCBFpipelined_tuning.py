from src.Base.Evaluation.Evaluator import EvaluatorHoldout
from src.GraphBased.P3alphaRecommender import P3alphaRecommender
from src.GraphBased.P3alphaCBFRecommender import P3alphaCBFRecommender
from src.Utils.load_ICM import load_ICM
from src.Utils.load_URM import load_URM
from src.Hybrid.SimilarityPipelinedHybridRecommender import SimilarityPipelinedHybridRecommender
import json
import os

URM_all = load_URM("../../in/data_train.csv")
ICM_all = load_ICM("../../in/data_ICM_title_abstract.csv")
from src.Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample

URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)

evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10], verbose=False)

recommender_1 = P3alphaRecommender(URM_train=URM_train, verbose=False)

with open('logs/' + recommender_1.RECOMMENDER_NAME + '_logs.json', 'r') as config_file:
    max = json.load(config_file)['params']

max['topK'] = int(max['topK'])

recommender_1.fit(**max, implicit=True)

recommender_2 = P3alphaCBFRecommender(URM_train=URM_train,ICM_train=ICM_all, verbose=False)

with open('logs/' + recommender_2.RECOMMENDER_NAME + '_logs.json', 'r') as config_file:
    max = json.load(config_file)['params']

max['topK'] = int(max['topK'])

recommender_2.fit(**max, implicit=False)

print(max)

recommender = SimilarityPipelinedHybridRecommender(URM_train=URM_train, recommender_1=recommender_1,recommender_2=recommender_2)
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
