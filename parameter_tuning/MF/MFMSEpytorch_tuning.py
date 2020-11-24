from src.Base.Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
from src.MatrixFactorization.PyTorch.MF_MSE_PyTorch import MF_MSE_PyTorch
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
        MF_MSE_PyTorch(URM_train=URMs_train[index])
    )


tuning_params = {
    "num_factors": (50, 500)
}

results = []

def BO_func(
        num_factors
):
    for index in range(len(recommenders)):
        recommenders[index].fit(
            epochs=1000,
            num_factors=int(num_factors),
            learning_rate=1e-3,
            batch_size=700,
            **{
                'epochs_min' : 0,
                'evaluator_object' : evaluator_validation.evaluator_list[index],
                'stop_on_validation' : True,
                'validation_every_n' : 10,
                'validation_metric' : 'MAP',
                'lower_validations_allowed' : 3
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
    n_iter=8
)

import json

with open("logs/" + recommenders[0].RECOMMENDER_NAME + "_logs.json", 'w') as json_file:
    json.dump(optimizer.max, json_file)

from src.Base.Evaluation.k_fold_significance_test import compute_k_fold_significance

for index in range(len(recommenders)):
    recommenders[index].fit(
            epochs=1000,
            num_factors=int(optimizer.max["params"]["num_factors"]),
            learning_rate=1e-3,
            batch_size=700,
            **{
                'epochs_min' : 0,
                'evaluator_object' : evaluator_validation.evaluator_list[index],
                'stop_on_validation' : True,
                'validation_every_n' : 10,
                'validation_metric' : 'MAP',
                'lower_validations_allowed' : 3
            }
    )

result = evaluator_validation.evaluateRecommender(recommenders)

compute_k_fold_significance(result, *results)
