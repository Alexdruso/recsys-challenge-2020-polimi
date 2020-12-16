from src.Base.Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
from src.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
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
        SLIM_BPR_Cython(URM_train=URMs_train[index])
    )


tuning_params = {
    "batch_size":(128, 1024),
    "lambda_i":(1e-5, 1e-2),
    "lambda_j":(1e-5, 1e-2),
    "topK":(100,1000),
    "epochs": (10, 100)
    #"gamma":(1e-5, 1e-2),
    #"beta_1":(1e-5, 1e-2),
    #"beta_2":(1e-5, 1e-2)
}

results = []

def BO_func(
        batch_size,
        lambda_i,
        lambda_j,
        topK,
        epochs
):
    for index in range(len(recommenders)):
        recommenders[index].fit(
            epochs=epochs,
            positive_threshold_BPR=None,
            train_with_sparse_weights=True,
            symmetric=False,
            random_seed=None,
            batch_size=batch_size,
            lambda_i=lambda_i,
            lambda_j=lambda_j,
            learning_rate=1e-4,
            topK=int(topK),
            sgd_mode='adagrad',
            #gamma=,
            #beta_1=,
            #beta_2=,
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
    n_iter=5
)

import json

with open("logs/" + recommenders[0].RECOMMENDER_NAME + "_logs.json", 'w') as json_file:
    json.dump(optimizer.max, json_file)

