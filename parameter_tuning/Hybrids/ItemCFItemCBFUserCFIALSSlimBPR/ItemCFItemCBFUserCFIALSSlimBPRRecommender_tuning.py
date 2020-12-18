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

for k in range(1):
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)
    URMs_train.append(URM_train)
    URMs_validation.append(URM_validation)

evaluator_validation = K_Fold_Evaluator_MAP(URMs_validation, cutoff_list=[10], verbose=False)

ICMs_combined = []
for URM in URMs_train:
    ICMs_combined.append(combine(ICM=ICM_all, URM=URM))

from src.Hybrid.GeneralizedMergedHybridRecommender import GeneralizedMergedHybridRecommender
from src.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from src.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from src.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from src.Utils.confidence_scaling import linear_scaling_confidence
from src.Implicit.ImplicitALSRecommender import ImplicitALSRecommender
from src.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython

from bayes_opt import BayesianOptimization

itemCBF_recommenders = []
itemCF_recommenders = []
userCF_recommenders = []
IALS_recommenders=[]
slimBPR_recommenders = []
recommenders = []

for index in range(len(URMs_train)):

    itemCF_recommenders.append(
        ItemKNNCFRecommender(
            URM_train=URMs_train[index],
            verbose=False
        )
    )

    itemCF_recommenders[index].fit(
        similarity="jaccard",
        topK=200,
        shrink=200,
        normalize=False
    )

    itemCBF_recommenders.append(
        ItemKNNCBFRecommender(
            URM_train=URMs_train[index],
            ICM_train=ICM_all,
            verbose=False
        )
    )

    itemCBF_recommenders[index].fit(
        similarity="jaccard",
        topK=200,
        shrink=10,
        normalize=False
    )

    userCF_recommenders.append(
        UserKNNCFRecommender(
            URM_train=URMs_train[index],
            verbose=False
        )
    )

    userCF_recommenders[index].fit(
        similarity="cosine",topK=524,shrink=52, normalize=False, feature_weighting="TF-IDF"
    )

    IALS_recommenders.append(
        ImplicitALSRecommender(
            URM_train=URMs_train[index],
            verbose=False
        )
    )

    IALS_recommenders[index].fit(
        factors=int(188.56989379722654),
        regularization=0.01,
        use_gpu=False,
        iterations=int(10),
        num_threads=4,
        confidence_scaling=linear_scaling_confidence,
        **{"alpha": 50}
    )

    slimBPR_recommenders.append(
        SLIM_BPR_Cython(
                URM_train=URMs_train[index],
            verbose=False
        )
    )

    slimBPR_recommenders[index].fit(
        epochs=98.33218638454095,
        positive_threshold_BPR=None,
        train_with_sparse_weights=True,
        symmetric=False,
        random_seed=None,
        batch_size=548.6615810947892,
        lambda_i=0.0039954585909894764,
        lambda_j=0.008139187451093313,
        learning_rate=1e-4,
        topK=int(591.8108481752299),
        sgd_mode='adagrad',
        # gamma=,
        # beta_1=,
        # beta_2=,
    )

    recommenders.append(
        GeneralizedMergedHybridRecommender(
        URM_train=URMs_train[index],
        recommenders=[
            itemCF_recommenders[index],
            itemCBF_recommenders[index],
            userCF_recommenders[index],
            IALS_recommenders[index],
            slimBPR_recommenders[index]
        ],
        verbose=False
    )
    )

import numpy as np

results = []

for weight1 in np.linspace(start=0,stop=1,num=10):
    for weight2 in np.linspace(start=0,stop=1,num=5):
        for weight3 in np.linspace(start=0, stop=1, num=5):
            for weight4 in np.linspace(start=0, stop=1, num=5):
                for weight5 in np.linspace(start=0,stop=1,num=5):
                    for index in range(len(URMs_train)):

                        recommenders[index].fit(
                            alphas=[
                                weight1,
                                weight2,
                                weight3,
                                weight4,
                                weight5
                            ]
                        )

                    result = evaluator_validation.evaluateRecommender(recommenders)

                    results.append(
                        {
                            'target': sum(result) / len(result),
                            'params': {
                                'alphas': [
                                    weight1,
                                    weight2,
                                    weight3,
                                    weight4,
                                    weight5
                                ]
                            }
                        }
                    )

                    print(results[-1])

import json

with open("logs/"+ recommenders[0].RECOMMENDER_NAME+"_logs.json", 'w') as json_file:
    json.dump(max(results, key= lambda x: x['target']), json_file)

