from src.Base.Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
from src.Utils.ICM_preprocessing import *
from src.Utils.confidence_scaling import *
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

from src.Hybrid.GeneralizedMergedHybridRecommender import GeneralizedMergedHybridRecommender
from src.Implicit.FeatureCombinedImplicitALSRecommender import FeatureCombinedImplicitALSRecommender
from src.GraphBased.UserRP3betaRecommender import UserRP3betaRecommender
from src.GraphBased.RP3betaCBFRecommender import RP3betaCBFRecommender

from bayes_opt import BayesianOptimization

IALS_recommenders = []
rp3betaCBF_recommenders = []
userRP3beta_recommenders = []

for index in range(len(URMs_train)):
    IALS_recommenders.append(
        FeatureCombinedImplicitALSRecommender(
            URM_train=URMs_train[index],
            ICM_train=ICM_all,
            verbose=True
        )
    )
    IALS_recommenders[index].fit(
        factors=int(250),
        regularization=0.01,
        use_gpu=False,
        iterations=43,
        num_threads=4,
        confidence_scaling=linear_scaling_confidence,
        **{
            'URM': {"alpha": 50},
            'ICM': {"alpha": 50}
        }
    )

    rp3betaCBF_recommenders.append(
        RP3betaCBFRecommender(
            URM_train=URMs_train[index],
            ICM_train=ICMs_combined[index],
            verbose=False
        )
    )

    rp3betaCBF_recommenders[index].fit(
        topK=int(529.1628484087545),
        alpha=0.45304737831676245,
        beta=0.226647894170121,
        implicit=False
    )

    userRP3beta_recommenders.append(
        UserRP3betaRecommender(
            URM_train=ICMs_combined[index].T,
            verbose=False
        )
    )

    userRP3beta_recommenders[index].fit(
        topK=int(252.2),
        alpha=0.2658,
        beta=0.2847,
        implicit=False
    )

tuning_params = {
    "hybridWeight1": (0, 1),
    "hybridWeight2": (0, 1),
    "hybridWeight3": (0, 1)
}

results = []


def BO_func(
        hybridWeight1,
        hybridWeight2,
        hybridWeight3
):
    recommenders = []

    for index in range(len(URMs_train)):

        recommender = GeneralizedMergedHybridRecommender(
            URM_train=URMs_train[index],
            recommenders=[
                IALS_recommenders[index],
                rp3betaCBF_recommenders[index],
                userRP3beta_recommenders[index]
            ],
            verbose=False
        )

        totalWeight = hybridWeight1+hybridWeight2+hybridWeight3
        recommender.fit(
            alphas=[
                hybridWeight1/totalWeight,
                hybridWeight2/totalWeight,
                hybridWeight3/totalWeight
            ]
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
    init_points=30,
    n_iter=50,
)

recommender = GeneralizedMergedHybridRecommender(
            URM_train=URMs_train[0],
            recommenders=[
                IALS_recommenders[0],
                rp3betaCBF_recommenders[0],
                userRP3beta_recommenders[0]
            ],
            verbose=False
        )
recommender.fit()

import json

with open("logs/FeatureCombined" + recommender.RECOMMENDER_NAME + "_logs.json", 'w') as json_file:
    json.dump(optimizer.max, json_file)
