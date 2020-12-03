from src.Base.Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
from src.Utils.load_ICM import load_ICM
from src.Utils.load_URM import load_URM
from src.Utils.ICM_preprocessing import *

URM_all = load_URM("../../../in/data_train.csv")
ICM_all = load_ICM("../../../in/data_ICM_title_abstract.csv")
from src.Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample

URMs_train = []
URMs_validation = []

for k in range(5):
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)
    URMs_train.append(URM_train)
    URMs_validation.append(URM_validation)

evaluator_validation = K_Fold_Evaluator_MAP(URMs_validation, cutoff_list=[10], verbose=False)

ICMs_combined = []
for URM in URMs_train:
    ICMs_combined.append(combine(ICM=ICM_all, URM=URM))

from src.GraphBased.RP3betaCBFRecommender import RP3betaCBFRecommender
from src.FeatureWeighting.CFW_D_Similarity_Linalg import CFW_D_Similarity_Linalg

from bayes_opt import BayesianOptimization

CFWD_recommenders = []

for index in range(len(URMs_train)):
    service_recommender = RP3betaCBFRecommender(
            URM_train=URMs_train[index],
            ICM_train=ICMs_combined[index],
            verbose=False
        )

    service_recommender.fit(topK=529, alpha=0.453, beta=0.227)

    CFWD_recommenders.append(
        CFW_D_Similarity_Linalg(
            URM_train=URMs_train[index],
            ICM=ICM_all,
            S_matrix_target=service_recommender.W_sparse
        )
    )

tuning_params = {
    "topK": (10, 1000),
    "add_zeros_quota": (0.0,1.0)
}

results = []


def BO_func(
        topK,
        add_zeros_quota
):
    for recommender in CFWD_recommenders:
        recommender.fit(topK=int(topK), add_zeros_quota= add_zeros_quota, show_max_performance=True)

    result = evaluator_validation.evaluateRecommender(CFWD_recommenders)
    results.append(result)
    return sum(result) / len(result)


optimizer = BayesianOptimization(
    f=BO_func,
    pbounds=tuning_params,
    verbose=5,
    random_state=5,
)

optimizer.maximize(
    init_points=30,
    n_iter=20,
)


import json

with open("logs/FeatureCombined" + CFWD_recommenders[0].RECOMMENDER_NAME + "_logs.json", 'w') as json_file:
    json.dump(optimizer.max, json_file)
