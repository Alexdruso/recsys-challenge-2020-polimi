from src.Base.Evaluation.Evaluator import EvaluatorHoldout
from src.Utils.ICM_preprocessing import *
from src.Utils.load_ICM import load_ICM
from src.Utils.load_URM import load_URM

URM_all = load_URM("../../in/data_train.csv")
ICM_all = load_ICM("../../in/data_ICM_title_abstract.csv")

binarize_ICM(ICM_all)
ICM_all = combine(ICM=ICM_all, URM=URM_all)
from src.Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample

URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)

evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10], verbose=False)


ICM_augmented = sps.hstack((URM_train.T, ICM_all), format='csr')

from src.Hybrid.SimilarityMergedHybridRecommender import SimilarityMergedHybridRecommender
from src.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from src.GraphBased.RP3betaCBFRecommender import RP3betaCBFRecommender

from bayes_opt import BayesianOptimization

easer_recommender = EASE_R_Recommender(URM_train=URM_train, sparse_threshold_quota=1.0)

easer_recommender.fit(topK=56, l2_norm=41476.92126107723)

rp3betaCBF_recommender = RP3betaCBFRecommender(URM_train=URM_train, ICM_train=ICM_augmented, verbose=False)

tuning_params = {
    "cbfTopK": (600, 700),
    "cbfAlpha": (0.3, 0.5),
    "cbfBeta": (0.1, 0.3),
    "hybridTopK": (10, 500),
    "hybridAlpha": (0.1, 0.9)
}


def BO_func(
        cbfTopK,
        cbfAlpha,
        cbfBeta,
        hybridTopK,
        hybridAlpha
):

    rp3betaCBF_recommender.fit(
        topK=int(cbfTopK),
        alpha=cbfAlpha,
        beta=cbfBeta,
        implicit=False
    )

    recommender = SimilarityMergedHybridRecommender(
        URM_train=URM_train,
        CFRecommender=easer_recommender,
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

easer_recommender = EASE_R_Recommender(URM_train=URM_all)

easer_recommender.fit(
)

rp3betaCBF_recommender = RP3betaCBFRecommender(URM_train=URM_all, ICM_train=ICM_all)

rp3betaCBF_recommender.fit(
    topK=int(hyperparameters['cbfTopK']),
    alpha=hyperparameters['cbfAlpha'],
    beta=hyperparameters['cbfBeta'],
    implicit=True,
    normalize_similarity=True
)

recommender = SimilarityMergedHybridRecommender(
    URM_train=URM_all,
    CFRecommender=easer_recommender,
    CBFRecommender=rp3betaCBF_recommender
)
recommender.fit(
    topK=int(hyperparameters['hybridTopK']),
    alpha=hyperparameters['hybridAlpha']
)


import json

with open("logs/FeatureAugmented" + recommender.RECOMMENDER_NAME + "_logs.json", 'w') as json_file:
    json.dump(optimizer.max, json_file)


#recommender.save_model(folder_path='../models/', file_name=recommender.RECOMMENDER_NAME)
