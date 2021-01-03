from src.Base.Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
from src.Implicit.FeatureCombinedImplicitALSRecommender import FeatureCombinedImplicitALSRecommender
from src.Utils.confidence_scaling import linear_scaling_confidence
from src.Utils.load_ICM import load_ICM
from src.Utils.load_URM import load_URM

URM_all = load_URM("../../in/data_train.csv")
ICM_all = load_ICM("../../in/data_ICM_title_abstract.csv")
from src.Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample

URMs_train = []
URMs_validation = []


for k in range(5):
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)
    URMs_train.append(URM_train)
    URMs_validation.append(URM_validation)

evaluator_validation = K_Fold_Evaluator_MAP(URMs_validation, cutoff_list=[10], verbose=False)


recommenders = []

for index in range(len(URMs_train)):
    recommenders.append(
        FeatureCombinedImplicitALSRecommender(
            URM_train=URMs_train[index],
            ICM_train=ICM_all,
            verbose=True
        )
    )


tuning_params = {
    "urm_alpha":(40, 100),
    "icm_alpha":(40, 100),
    "factors":(200,400),
    "epochs": (10, 100)
}

results = []

def BO_func(
        factors,
        epochs,
        urm_alpha,
        icm_alpha
):
    for index in range(len(recommenders)):
        recommenders[index].fit(
            factors=int(factors),
            regularization=0.01,
            use_gpu=False,
            iterations=int(epochs),
            num_threads=4,
            confidence_scaling=linear_scaling_confidence,
            **{
                'URM' : {"alpha":urm_alpha},
                'ICM': {"alpha": icm_alpha}
            }
        )

    result = evaluator_validation.evaluateRecommender(recommenders)
    results.append(result)
    return sum(result) / len(result)


from bayes_opt import BayesianOptimization

optimizer = BayesianOptimization(
    f=BO_func,
    pbounds=tuning_params,
    verbose=5,
    random_state=5,
)

optimizer.maximize(
    init_points=10,
    n_iter=20
)

import json

with open("logs/" + recommenders[0].RECOMMENDER_NAME + "_logs.json", 'w') as json_file:
    json.dump(optimizer.max, json_file)

