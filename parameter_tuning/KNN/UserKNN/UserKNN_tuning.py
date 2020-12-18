from src.Base.Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
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

from src.KNN.UserKNNCFRecommender import UserKNNCFRecommender

tuning_params = {
    "topK": (10, 600),
    "shrink": (10, 200)
}

userKNN_recommenders = []

for index in range(len(URMs_train)):
    userKNN_recommenders.append(
        UserKNNCFRecommender(
            URM_train=URMs_train[index],
            verbose=False
        )
    )

for similarity in ['cosine', 'pearson', 'jaccard', 'tanimoto', 'adjusted']:
    for feature_weighting in ["BM25", "TF-IDF", "none"]:

        print("Optimizing for similarity {} and feature weighting {}...".format(similarity, feature_weighting))
        def BO_func(topK, shrink):
            for recommender in userKNN_recommenders:
                recommender.fit(
                    topK=topK,
                    shrink=shrink,
                    similarity=similarity,
                    feature_weighting=feature_weighting,
                    normalize=False
                )

            result = evaluator_validation.evaluateRecommender(userKNN_recommenders)
            return sum(result) / len(result)


        from bayes_opt import BayesianOptimization

        optimizer = BayesianOptimization(
            f=BO_func,
            pbounds=tuning_params,
            verbose=5,
            random_state=5,
        )

        optimizer.maximize(
            init_points=5,
            n_iter=3,
        )

        import json

        with open("logs/" + userKNN_recommenders[0].RECOMMENDER_NAME + "_{}_{}_logs.json".format(similarity, feature_weighting),
                  'w') as json_file:
            json.dump(optimizer.max, json_file)
