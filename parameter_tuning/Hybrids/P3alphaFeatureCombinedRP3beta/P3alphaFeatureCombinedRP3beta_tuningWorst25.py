from src.Base.Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
from src.Utils.ICM_preprocessing import *
from src.Utils.load_ICM import load_ICM
from src.Utils.load_URM import load_URM

URM_all = load_URM("../../../in/data_train.csv")
ICM_all = load_ICM("../../../in/data_ICM_title_abstract.csv")
from src.Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample

URMs_train = []
URMs_validation = []
ignore_users_list = []

import numpy as np

for k in range(5):
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)
    URMs_train.append(URM_train)
    URMs_validation.append(URM_validation)

    profile_length = np.ediff1d(URM_train.indptr)
    block_size = int(len(profile_length) * 0.25)

    start_pos = 0 * block_size
    end_pos = min(1 * block_size, len(profile_length))
    sorted_users = np.argsort(profile_length)

    users_in_group = sorted_users[start_pos:end_pos]

    users_in_group_p_len = profile_length[users_in_group]
    sorted_users = np.argsort(profile_length)

    users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
    ignore_users_list.append(sorted_users[users_not_in_group_flag])

evaluator_validation = K_Fold_Evaluator_MAP(URMs_validation, cutoff_list=[10], verbose=False,
                                            ignore_users_list=ignore_users_list)

ICMs_combined = []
for URM in URMs_train:
    ICMs_combined.append(combine(ICM=ICM_all, URM=URM))

from src.Hybrid.SimilarityMergedHybridRecommender import SimilarityMergedHybridRecommender
from src.GraphBased.P3alphaRecommender import P3alphaRecommender
from src.GraphBased.RP3betaCBFRecommender import RP3betaCBFRecommender

from bayes_opt import BayesianOptimization

p3alpha_recommenders = []
rp3betaCBF_recommenders = []

for index in range(len(URMs_train)):
    p3alpha_recommenders.append(
        P3alphaRecommender(
            URM_train=URMs_train[index],
            verbose=False
        )
    )

    rp3betaCBF_recommenders.append(
        RP3betaCBFRecommender(
            URM_train=URMs_train[index],
            ICM_train=ICMs_combined[index],
            verbose=False
        )
    )

tuning_params = {
    "cfTopK": (210, 230),
    "cfAlpha": (0.45, 0.52),
    "cbfTopK": (500, 600),
    "cbfAlpha": (0.4, 0.5),
    "cbfBeta": (0.2, 0.3),
    "hybridTopK": (10, 500),
    "hybridAlpha": (0.1, 0.9)
}

results = []


def BO_func(
        cfTopK,
        cfAlpha,
        cbfTopK,
        cbfAlpha,
        cbfBeta,
        hybridTopK,
        hybridAlpha
):
    recommenders = []

    for index in range(len(URMs_train)):
        p3alpha_recommenders[index].fit(
            topK=int(cfTopK),
            alpha=cfAlpha,
            implicit=True
        )

        rp3betaCBF_recommenders[index].fit(
            topK=int(cbfTopK),
            alpha=cbfAlpha,
            beta=cbfBeta,
            implicit=False
        )

        recommender = SimilarityMergedHybridRecommender(
            URM_train=URMs_train[index],
            CFRecommender=p3alpha_recommenders[index],
            CBFRecommender=rp3betaCBF_recommenders[index],
            verbose=False
        )

        recommender.fit(
            topK=int(hybridTopK),
            alpha=hybridAlpha
        )

        recommenders.append(recommender)

    result = evaluator_validation.evaluateRecommender(recommenders)
    results.append(result)
    return sum(result) / len(result)


optimizer = BayesianOptimization(
    f=BO_func,
    pbounds=tuning_params,
    verbose=5,
    random_state=5,
)

optimizer.maximize(
    init_points=10,
    n_iter=18,
)

hyperparameters = optimizer.max['params']

p3alpha_recommender = P3alphaRecommender(URM_train=URM_all)

p3alpha_recommender.fit(
    topK=int(hyperparameters['cfTopK']),
    alpha=hyperparameters['cfAlpha'],
    implicit=True,
    normalize_similarity=True
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
    CFRecommender=p3alpha_recommender,
    CBFRecommender=rp3betaCBF_recommender
)
recommender.fit(
    topK=int(hyperparameters['hybridTopK']),
    alpha=hyperparameters['hybridAlpha']
)

import json

with open("logs/P3alphaFeatureCombinedRP3beta_tuning_logsWorst25.json", 'w') as json_file:
    json.dump(optimizer.max, json_file)

from src.Base.Evaluation.k_fold_significance_test import compute_k_fold_significance

recommenders = []

for index in range(len(URMs_train)):
    p3alpha_recommenders[index].fit(
        topK=int(optimizer.max["params"]["cfTopK"]),
        alpha=optimizer.max["params"]["cfAlpha"],
        implicit=True
    )

    rp3betaCBF_recommenders[index].fit(
        topK=int(optimizer.max["params"]["cbfTopK"]),
        alpha=optimizer.max["params"]["cbfAlpha"],
        beta=optimizer.max["params"]["cbfBeta"],
        implicit=False
    )

    recommender = SimilarityMergedHybridRecommender(
        URM_train=URMs_train[index],
        CFRecommender=p3alpha_recommenders[index],
        CBFRecommender=rp3betaCBF_recommenders[index],
        verbose=False
    )

    recommender.fit(
        topK=int(optimizer.max["params"]["hybridTopK"]),
        alpha=optimizer.max["params"]["hybridAlpha"]
    )

    recommenders.append(recommender)

result = evaluator_validation.evaluateRecommender(recommenders)

compute_k_fold_significance(result, *results)
