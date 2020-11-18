from src.Base.Evaluation.Evaluator import EvaluatorHoldout
from src.Utils.load_ICM import load_ICM
from src.Utils.load_URM import load_URM

URM_all = load_URM("../in/data_train.csv")
ICM_all = load_ICM("../in/data_ICM_title_abstract.csv")
from src.Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample

URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)

evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10], verbose=False)

from src.Hybrid.SimilarityMergedHybridRecommender import SimilarityMergedHybridRecommender
from src.GraphBased.P3alphaRecommender import P3alphaRecommender
from src.GraphBased.P3alphaCBFRecommender import P3alphaCBFRecommender

from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

from bayes_opt.util import load_logs

from bayes_opt import BayesianOptimization

p3alpha_recommender = P3alphaRecommender(URM_train=URM_train)

p3alphaCBF_recommender = P3alphaCBFRecommender(URM_train=URM_train, ICM_train=ICM_all)


tuning_params = {
    "cfTopK":(10, 500),
    "cfAlpha":(0.1, 1),
    "cbfTopK":(10, 500),
    "cbfAlpha":(0.1,0.9),
    "hybridTopK": (10, 500),
    "hybridAlpha": (0.1, 0.9)
}


def BO_func(hybridTopK, alpha):
    recommender = SimilarityMergedHybridRecommender(
        URM_train=URM_train,
        CFRecommender=p3alpha_recommender,
        CBFRecommender=p3alphaCBF_recommender,
        verbose=False
    )

    recommender.fit(topK=int(hybridTopK), alpha=alpha)
    result_dict, _ = evaluator_validation.evaluateRecommender(recommender)

    return result_dict[10]["MAP"]


optimizer = BayesianOptimization(
    f=BO_func,
    pbounds=tuning_params,
    verbose=5,
    random_state=5,
)

logger = JSONLogger(path="logs/" + recommender.RECOMMENDER_NAME + "_logs.json")
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

optimizer.maximize(
    init_points=5,
    n_iter=40,
)

hyperparameters = optimizer.max['params']

p3alpha_recommender = P3alphaRecommender(URM_train=URM_all)

p3alphaCBF_recommender = P3alphaCBFRecommender(URM_train=URM_all, ICM_train=ICM_all)

recommender = SimilarityMergedHybridRecommender(
    URM_train=URM_all,
    CFRecommender=p3alpha_recommender,
    CBFRecommender=p3alphaCBF_recommender
)
recommender.fit(topK=int(hyperparameters['hybridTopK']), alpha=hyperparameters['hybridAlpha'])

recommender.save_model(folder_path='../models/', file_name=recommender.RECOMMENDER_NAME)
