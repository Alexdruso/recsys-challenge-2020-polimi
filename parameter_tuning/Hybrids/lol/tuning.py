from src.Base.Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
from src.Utils.ICM_preprocessing import *
from src.Utils.load_ICM import load_ICM
from src.Utils.load_URM import load_URM
import sklearn

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
from src.Hybrid.GeneralizedMergedHybridRecommender import GeneralizedMergedHybridRecommender
from src.GraphBased.P3alphaRecommender import P3alphaRecommender
from src.GraphBased.RP3betaCBFRecommender import RP3betaCBFRecommender
from src.GraphBased.UserRP3betaRecommender import UserRP3betaRecommender
from src.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from src.MatrixFactorization.PureSVDRecommender import PureSVDItemRecommender
from src.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython

from bayes_opt import BayesianOptimization

p3alpha_recommenders = []
rp3betaCBF_recommenders = []
rp3betaCombined_recommenders = []
userRp3beta_recommender = []
itemKNN_recommenders = []
pureSVD_recommenders = []
slimBPRCombined_recommenders = []
recommenders = []

for index in range(len(URMs_train)):

    p3alpha_recommenders.append(
        P3alphaRecommender(
            URM_train=URMs_train[index],
            verbose=False
        )
    )

    p3alpha_recommenders[index].fit(
        topK=int(211.19325949126622),
        alpha=0.5011972162287313,
        implicit=True
    )

    # rp3betaCombined_recommenders.append(
    #     RP3betaCBFRecommender(
    #         URM_train=URMs_train[index],
    #         ICM_train=ICMs_combined[index],
    #         verbose=False
    #     )
    # )
    #
    # rp3betaCombined_recommenders[index].fit(
    #     topK=int(525.3588205773788),
    #     alpha=0.42658191175355076,
    #     beta=0.2284685880641364,
    #     implicit=False
    # )

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

    userRp3beta_recommender.append(
        UserRP3betaRecommender(
            URM_train=URMs_train[index],
            verbose=False
        )
    )

    userRp3beta_recommender[index].fit(
        topK=int(218.06232565392185),
        alpha=0.34942741182485626,
        beta=0.5381272103966592,
        implicit=True
    )

    # itemKNN_recommenders.append(
    #     ItemKNNCFRecommender(
    #         URM_train=URMs_train[index],
    #         verbose=False
    #     )
    # )
    #
    # itemKNN_recommenders[index].fit(
    #     topK=100,
    #     shrink=50
    # )

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

    slimBPRCombined_recommenders.append(
        SLIM_BPR_Cython(
                URM_train=ICMs_combined[index].T,
            verbose=False
        )
    )

    slimBPRCombined_recommenders[index].fit(
        epochs=39.54,
        positive_threshold_BPR=None,
        train_with_sparse_weights=True,
        symmetric=False,
        random_seed=None,
        batch_size=393.93324941229486,
        lambda_i=0.004419,
        lambda_j=0.001592,
        learning_rate=1e-4,
        topK=int(891.9),
        sgd_mode='adagrad',
        # gamma=,
        # beta_1=,
        # beta_2=,
    )



    slimBPRCombined_recommenders[index].URM_train = URMs_train[index]

    recommenders.append(
        GeneralizedMergedHybridRecommender(
        URM_train=URMs_train[index],
        recommenders=[
            p3alpha_recommenders[index],
            # rp3betaCombined_recommenders[index],
            rp3betaCBF_recommenders[index],
            # itemKNN_recommenders[index],
            userRp3beta_recommender[index],
            pureSVD_recommenders[index],
            slimBPRCombined_recommenders[index]
        ],
        verbose=False
    )
    )
tuning_params = {
    "weight1": (0, 1),
    "weight2": (0, 1),
    "weight3": (0, 1),
    "weight4": (0, 1),
    "weight5": (0, 1),
    # "weight6": (0, 1)
}

results = []
def BO_func(
        weight1,
        weight2,
        weight3,
        weight4,
        weight5,
        # weight6
):

    for index in range(len(URMs_train)):

        recommenders[index].fit(
            alphas=[
                weight1,
                weight2,
                weight3,
                weight4,
                weight5,
                # weight6
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
    init_points=100,
    n_iter=200,
)

import json

with open("logs/"+ recommenders[0].RECOMMENDER_NAME+"_logs.json", 'w') as json_file:
    json.dump(optimizer.max, json_file)
