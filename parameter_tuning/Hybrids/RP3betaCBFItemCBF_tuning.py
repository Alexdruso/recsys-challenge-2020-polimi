from src.Base.Evaluation.Evaluator import EvaluatorHoldout
from src.Utils.load_ICM import load_ICM
from src.Utils.load_URM import load_URM

URM_all = load_URM("../../in/data_train.csv")
ICM_all = load_ICM("../../in/data_ICM_title_abstract.csv")
from src.Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample

URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)

evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10], verbose=False)

from src.Hybrid.SimilarityMergedHybridRecommender import SimilarityMergedHybridRecommender
from src.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from src.GraphBased.RP3betaCBFRecommender import RP3betaCBFRecommender

from bayes_opt import BayesianOptimization

itemKNNCBF_recommender = ItemKNNCBFRecommender(URM_train=URM_train, ICM_train=ICM_all, verbose=False)

rp3betaCBF_recommender = RP3betaCBFRecommender(URM_train=URM_train, ICM_train=ICM_all, verbose=False)

tuning_params = {
    "cfTopK": (300, 400),
    "cfShrink": (10, 20),
    "cbfTopK": (10, 100),
    "cbfAlpha": (0.2, 0.3),
    "cbfBeta": (0.3,0.4),
    "hybridTopK": (10, 500),
    "hybridAlpha": (0.1, 0.9)
}


def BO_func(
        cfTopK,
        cfShrink,
        cbfTopK,
        cbfAlpha,
        cbfBeta,
        hybridTopK,
        hybridAlpha
):
    itemKNNCBF_recommender.fit(
        topK=int(cfTopK),
        shrink=int(cfShrink),
        similarity='jaccard'
    )

    rp3betaCBF_recommender.fit(
        topK=int(cbfTopK),
        alpha=cbfAlpha,
        beta=cbfBeta,
        implicit=False
    )

    recommender = SimilarityMergedHybridRecommender(
        URM_train=URM_train,
        CFRecommender=itemKNNCBF_recommender,
        CBFRecommender=rp3betaCBF_recommender,
        verbose=False
    )

    recommender.fit(
        topK=int(hybridTopK),
        alpha=hybridAlpha
    )
    result_dict, _ = evaluator_validation.evaluateRecommender(recommender)

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

hyperparameters = optimizer.max['params']

itemKNNCBF_recommender = ItemKNNCBFRecommender(URM_train=URM_all, ICM_train=ICM_all)

itemKNNCBF_recommender.fit(
    topK=int(hyperparameters['cfTopK']),
    alpha=hyperparameters['cfShrink'],
    similarity='jaccard'
)

rp3betaCBF_recommender = RP3betaCBFRecommender(URM_train=URM_all, ICM_train=ICM_all)

rp3betaCBF_recommender.fit(
    topK=int(hyperparameters['cbfTopK']),
    alpha=hyperparameters['cbfAlpha'],
    beta=hyperparameters['cbfBeta'],
    implicit=True
)

recommender = SimilarityMergedHybridRecommender(
    URM_train=URM_all,
    CFRecommender=itemKNNCBF_recommender,
    CBFRecommender=rp3betaCBF_recommender
)
recommender.fit(
    topK=int(hyperparameters['hybridTopK']),
    alpha=hyperparameters['hybridAlpha']
)

import json

with open("logs/" + recommender.RECOMMENDER_NAME + "_logs.json", 'w') as json_file:
    json.dump(optimizer.max, json_file)


#recommender.save_model(folder_path='../models/', file_name=recommender.RECOMMENDER_NAME)
