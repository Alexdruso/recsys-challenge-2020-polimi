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
from src.Hybrid.MergedHybridRecommender import MergedHybridRecommender
from src.GraphBased.P3alphaRecommender import P3alphaRecommender
from src.GraphBased.RP3betaCBFRecommender import RP3betaCBFRecommender
from src.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from src.MatrixFactorization.PureSVDRecommender import PureSVDItemRecommender
from src.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython

from bayes_opt import BayesianOptimization

rp3betaCombined_recommenders = []
slimBPRCombined_recommenders = []
recommenders = []

for index in range(len(URMs_train)):

    rp3betaCombined_recommenders.append(
        RP3betaCBFRecommender(
            URM_train=URMs_train[index],
            ICM_train=ICMs_combined[index],
            verbose=False
        )
    )

    rp3betaCombined_recommenders[index].fit(
        topK=int(529.1628484087545),
        alpha=0.45304737831676245,
        beta=0.226647894170121,
        implicit=False
    )

    slimBPRCombined_recommenders.append(
        SLIM_BPR_Cython(
            URM_train=URMs_train[index],
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
    slimBPRCombined_recommenders[index].W_sparse = sklearn.preprocessing.normalize(slimBPRCombined_recommenders[index].W_sparse, norm='l2', axis=1, copy=False, return_norm=False)

    recommenders.append(
        GeneralizedSimilarityMergedHybridRecommender(
        URM_train=URMs_train[index],
        similarityRecommenders=[
            rp3betaCombined_recommenders[index],
            slimBPRCombined_recommenders[index]
        ],
        verbose=False
    )
    )
tuning_params = {
    "hybrid1TopK": (10, 1000),
    "hybrid1Alpha": (0, 1)
}

results = []
def BO_func(
        hybrid1TopK,
        hybrid1Alpha
):

    for index in range(len(URMs_train)):

        recommenders[index].fit(
            topKs=[
                int(hybrid1TopK),
                ],
            alphas=[
                hybrid1Alpha,
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

with open("logs/"+ recommenders[0].RECOMMENDER_NAME+"_logs.json", 'w') as json_file:
    json.dump(optimizer.max, json_file)

