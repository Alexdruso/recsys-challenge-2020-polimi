from src.Base.Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
from src.Utils.load_URM import load_URM

URM_all = load_URM("../../../in/data_train.csv")
from src.Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample

URMs_train = []
URMs_validation = []

for k in range(5):
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)
    URMs_train.append(URM_train)
    URMs_validation.append(URM_validation)

evaluator_validation = K_Fold_Evaluator_MAP(URMs_validation, cutoff_list=[10], verbose=False)

from src.GraphBased.UserRP3betaRecommender import UserRP3betaRecommender

from bayes_opt import BayesianOptimization

userRP3beta_recommenders = []

for index in range(len(URMs_train)):
    userRP3beta_recommenders.append(
        UserRP3betaRecommender(
            URM_train=URMs_train[index],
            verbose=False
        )
    )

tuning_params = {
    "alpha": (0, 1),
    "beta": (0, 1),
    "topK": (10, 1000)
}

results = []


def BO_func(
        alpha,
        beta,
        topK
):
    for recommender in userRP3beta_recommenders:
        recommender.fit(alpha=alpha, beta=beta, topK=int(topK), implicit=True)

    result = evaluator_validation.evaluateRecommender(userRP3beta_recommenders)
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


import json

with open("logs/" + userRP3beta_recommenders[0].RECOMMENDER_NAME + "_logs.json", 'w') as json_file:
    json.dump(optimizer.max, json_file)
