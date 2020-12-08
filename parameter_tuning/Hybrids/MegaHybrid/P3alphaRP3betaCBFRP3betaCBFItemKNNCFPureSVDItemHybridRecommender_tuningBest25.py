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

    start_pos = 3 * block_size
    end_pos = min(4 * block_size, len(profile_length))
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

from src.Hybrid.GeneralizedSimilarityMergedHybridRecommender import GeneralizedSimilarityMergedHybridRecommender
from src.Hybrid.MergedHybridRecommender import MergedHybridRecommender
from src.GraphBased.P3alphaRecommender import P3alphaRecommender
from src.GraphBased.RP3betaCBFRecommender import RP3betaCBFRecommender
from src.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from src.MatrixFactorization.PureSVDRecommender import PureSVDItemRecommender

from bayes_opt import BayesianOptimization

p3alpha_recommenders = []
rp3betaCBF_recommenders = []
rp3betaCombined_recommenders = []
itemKNN_recommenders = []
pureSVD_recommenders = []
recommenders = []

for index in range(len(URMs_train)):

    p3alpha_recommenders.append(
        P3alphaRecommender(
            URM_train=URMs_train[index],
            verbose=False
        )
    )

    p3alpha_recommenders[index].fit(
        topK=int(212.8832860130684),
        alpha=0.4729294763382114,
        implicit=True
    )

    rp3betaCombined_recommenders.append(
        RP3betaCBFRecommender(
            URM_train=URMs_train[index],
            ICM_train=ICMs_combined[index],
            verbose=False
        )
    )

    rp3betaCombined_recommenders[index].fit(
        topK=int(525.3588205773788),
        alpha=0.42658191175355076,
        beta=0.2284685880641364,
        implicit=False
    )

    rp3betaCBF_recommenders.append(
        RP3betaCBFRecommender(
            URM_train=URMs_train[index],
            ICM_train=ICM_all,
            verbose=False
        )
    )

    rp3betaCBF_recommenders[index].fit(
        topK=int(58.16144182493173),
        alpha=0.26520214286626453,
        beta=0.36456352256640157,
        implicit=False
    )

    itemKNN_recommenders.append(
        ItemKNNCFRecommender(
            URM_train=URMs_train[index],
            verbose=False
        )
    )

    itemKNN_recommenders[index].fit(
        topK=100,
        shrink=50
    )

    pureSVD_recommenders.append(
        PureSVDItemRecommender(
            URM_train=URMs_train[index],
            verbose=False
        )
    )

    pureSVD_recommenders[index].fit(
        num_factors=772,
        topK= 599
    )

    recommenders.append(
        GeneralizedSimilarityMergedHybridRecommender(
        URM_train=URMs_train[index],
        similarityRecommenders=[
            p3alpha_recommenders[index],
            rp3betaCombined_recommenders[index],
            rp3betaCBF_recommenders[index],
            itemKNN_recommenders[index],
            pureSVD_recommenders[index]
        ],
        verbose=False
    )
    )
tuning_params = {
    "hybrid1TopK": (10, 738),
    "hybrid1Alpha": (0, 1),
    "hybrid2TopK": (10, 796),
    "hybrid2Alpha": (0, 1),
    "hybrid3TopK": (10, 896),
    "hybrid3Alpha": (0, 1),
    "hybrid4TopK": (10, 1495),
    "hybrid4Alpha": (0, 1)
}

results = []
def BO_func(
        hybrid1TopK,
        hybrid1Alpha,
        hybrid2TopK,
        hybrid2Alpha,
        hybrid3TopK,
        hybrid3Alpha,
        hybrid4TopK,
        hybrid4Alpha
):

    for index in range(len(URMs_train)):

        recommenders[index].fit(
            topKs=[
                int(hybrid1TopK),
                int(hybrid2TopK),
                int(hybrid3TopK),
                int(hybrid4TopK)
                ],
            alphas=[
                hybrid1Alpha,
                hybrid2Alpha,
                hybrid3Alpha,
                hybrid4Alpha
            ]
        )

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
    init_points=50,
    n_iter=50,
)

import json

with open("logs/"+ recommenders[0].RECOMMENDER_NAME+"_best25_logs.json", 'w') as json_file:
    json.dump(optimizer.max, json_file)

