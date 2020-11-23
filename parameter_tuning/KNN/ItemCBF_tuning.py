import numpy as np

from src.Base.Evaluation.Evaluator import EvaluatorHoldout
from src.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from src.Utils.load_ICM import load_ICM
from src.Utils.load_URM import load_URM

URM_all = load_URM("../../in/data_train.csv")
ICM_all = load_ICM("../../in/data_ICM_title_abstract.csv")
from src.Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample

URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)

evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10], verbose=False)

tuning_params = {
    "topK": (10, 500),
    "shrink": (0, 200)
}

recommender = ItemKNNCBFRecommender(URM_train=URM_train, ICM_train=ICM_all, verbose=False)

hyperparameters = {'target': 0.0}

for similarity in ['cosine', 'pearson', 'jaccard', 'tanimoto', 'adjusted', 'euclidean']:
    for feature_weighting in ["BM25", "TF-IDF", "none"]:

        def BO_func(topK, shrink):
            recommender.fit(topK=int(topK), shrink=shrink, similarity=similarity, feature_weighting=feature_weighting)
            result_dict, _ = evaluator_validation.evaluateRecommender(recommender)

            return result_dict[10]["MAP"]


        from bayes_opt import BayesianOptimization

        optimizer = BayesianOptimization(
            f=BO_func,
            pbounds=tuning_params,
            verbose=5,
            random_state=5,
        )

        from bayes_opt.logger import JSONLogger
        from bayes_opt.event import Events

        logger = JSONLogger(
            path="logs/" + recommender.RECOMMENDER_NAME + '_' + similarity + '_' + feature_weighting + "_logs.json"
        )
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

        optimizer.maximize(
            init_points=10,
            n_iter=10,
        )

        if optimizer.max['target'] > hyperparameters['target']:
            hyperparameters = optimizer.max
            hyperparameters['params']['similarity'] = similarity
            hyperparameters['params']['feature_weighting'] = feature_weighting
            print(hyperparameters)

hyperparameters = hyperparameters['params']
recommender = ItemKNNCBFRecommender(URM_train=URM_all, verbose=False)
recommender.fit(
    topK=int(hyperparameters['topK']),
    shrink=hyperparameters['shrink'],
    similarity=hyperparameters['similarity'],
    feature_weighting=hyperparameters['feature_weighting']
)

recommender.save_model(folder_path='../../models/', file_name=recommender.RECOMMENDER_NAME + '_' + hyperparameters['similarity'] + '_' + hyperparameters['feature_weighting'])
