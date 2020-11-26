from src.Base.Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
from src.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython
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


recommenders = []

for index in range(len(URMs_train)):
    recommenders.append(
        MatrixFactorization_BPR_Cython(URM_train=URMs_train[index])
    )


tuning_params = {
    "batch_size": (1,1024),
    "num_factors": (50, 500),
    "positive_reg": (1e-5,1e-2),
    "negative_reg": (1e-5,1e-2)
}

results = []

def BO_func(
        batch_size,
        num_factors,
        positive_reg,
        negative_reg
):
    for index in range(len(recommenders)):
        recommenders[index].fit(
            epochs=1500,
            batch_size=batch_size,
            num_factors=num_factors,
            positive_threshold_BPR=None,
            learning_rate=0.001,
            use_bias=False,
            sgd_mode='adam',
            negative_interactions_quota=0.0,
            init_mean=0.0, init_std_dev=0.1,
            user_reg=0.0, item_reg=0.0, bias_reg=0.0, positive_reg=positive_reg, negative_reg=negative_reg,
            random_seed=None,
            **{
                'epochs_min' : 0,
                'evaluator_object' : evaluator_validation.evaluator_list[index],
                'stop_on_validation' : True,
                'validation_every_n' : 5,
                'validation_metric' : 'MAP',
                'lower_validations_allowed' : 5
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
    init_points=20,
    n_iter=30
)

import json

with open("logs/" + recommenders[0].RECOMMENDER_NAME + "_logs.json", 'w') as json_file:
    json.dump(optimizer.max, json_file)

