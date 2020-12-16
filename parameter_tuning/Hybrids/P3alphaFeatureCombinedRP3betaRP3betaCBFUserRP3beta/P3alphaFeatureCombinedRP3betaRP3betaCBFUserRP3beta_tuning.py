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

for k in range(5):
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)
    URMs_train.append(URM_train)
    URMs_validation.append(URM_validation)

evaluator_validation = K_Fold_Evaluator_MAP(URMs_validation, cutoff_list=[10], verbose=False)

ICMs_combined = []
for URM in URMs_train:
    ICMs_combined.append(combine(ICM=ICM_all, URM=URM))

from src.Hybrid.GeneralizedSimilarityMergedHybridRecommender import GeneralizedSimilarityMergedHybridRecommender
from src.Hybrid.MergedHybridRecommender import MergedHybridRecommender
from src.GraphBased.P3alphaRecommender import P3alphaRecommender
from src.GraphBased.RP3betaCBFRecommender import RP3betaCBFRecommender
from src.GraphBased.UserRP3betaRecommender import UserRP3betaRecommender

from bayes_opt import BayesianOptimization

p3alpha_recommenders = []
rp3betaCBF_recommenders = []
rp3betaCombined_recommenders = []
userRp3beta_recommenders = []
similarity_recommenders = []
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
        topK=int(188.6),
        alpha=0.1324,
        beta=0.981,
        implicit=False
    )

    similarity_recommenders.append(
        GeneralizedSimilarityMergedHybridRecommender(
            URM_train=URMs_train[index],
            similarityRecommenders=[
                p3alpha_recommenders[index],
                rp3betaCombined_recommenders[index],
                rp3betaCBF_recommenders[index]
            ],
            verbose=False
        )
    )

    similarity_recommenders[index].fit(
        topKs=[
            int(482.3259592432915),
            int(872.7)
        ],
        alphas=[
            0.2324902889610141,
            0.7876
        ]
    )

    userRp3beta_recommenders.append(
        UserRP3betaRecommender(
            URM_train=URMs_train[index],
            verbose=False
        )
    )
tuning_params = {
    "hybrid3Alpha": (0, 1),
    "userAlpha": (0, 1),
    "userBeta": (0, 1),
    "userTopK": (10, 600)
}

results = []


def BO_func(
        userAlpha,
        userBeta,
        userTopK,
        hybrid3Alpha,
):
    recommenders = []

    for index in range(len(URMs_train)):
        userRp3beta_recommenders[index].fit(
            topK=int(userTopK),
            alpha=userAlpha,
            beta=userBeta,
            implicit=True
        )

        recommenders.append(
            MergedHybridRecommender(
                URM_train=URMs_train[index],
                recommender1= similarity_recommenders[index],
                recommender2= userRp3beta_recommenders[index],
                verbose=False
            )
        )

        recommenders[index].fit(
            alpha=hybrid3Alpha
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

recommender = GeneralizedSimilarityMergedHybridRecommender(
    URM_train=URMs_train[0],
    similarityRecommenders=[
        p3alpha_recommenders[0],
        rp3betaCombined_recommenders[0],
        rp3betaCBF_recommenders[0],
        userRp3beta_recommenders[0]
    ],
    verbose=False
)

with open("logs/" + recommender.RECOMMENDER_NAME + "_logs.json", 'w') as json_file:
    json.dump(optimizer.max, json_file)
