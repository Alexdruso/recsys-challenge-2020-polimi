from src.Base.Evaluation.Evaluator import EvaluatorHoldout
from src.Utils.load_ICM import load_ICM
from src.Utils.load_URM import load_URM

URM_all = load_URM("../../in/data_train.csv")
ICM_all = load_ICM("../../in/data_ICM_title_abstract.csv")
from src.Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample

URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)

evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10], verbose=False)

from src.Hybrid.SimilarityMergedHybridRecommender import SimilarityMergedHybridRecommender
from src.GraphBased.P3alphaRecommender import P3alphaRecommender
from src.MatrixFactorization.PureSVDRecommender import PureSVDItemRecommender

from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

from bayes_opt.util import load_logs

from bayes_opt import BayesianOptimization

p3alpha_optimizer = BayesianOptimization(
    f=None,
    pbounds={
    "topK": (10, 500),
    "alpha": (0.1, 0.9)
    }
)

pure_SVD_optimizer = BayesianOptimization(
    f=None,
    pbounds={
    "topK": (10, 500),
    "num_factors": (10, 500)
    }
)


p3alpha_recommender = P3alphaRecommender(URM_train=URM_train)

pure_SVD_recommender = PureSVDItemRecommender(URM_train=URM_train)

# New optimizer is loaded with previously seen points
load_logs(p3alpha_optimizer, logs=["logs/"+p3alpha_recommender.RECOMMENDER_NAME+'_logs.json'])

print(p3alpha_optimizer.max)

hyperparameters = p3alpha_optimizer.max['params']

p3alpha_recommender.fit(topK=int(hyperparameters['topK']), alpha=hyperparameters['alpha'], implicit=True)

load_logs(pure_SVD_optimizer, logs=["logs/" + pure_SVD_recommender.RECOMMENDER_NAME + '_logs.json'])

print(pure_SVD_optimizer.max)

hyperparameters =pure_SVD_optimizer.max['params']

pure_SVD_recommender.fit(
    topK=int(hyperparameters['topK']),
    num_factors=int(hyperparameters['num_factors']),
)


tuning_params = {
    "topK": (10, 500),
    "alpha": (0.3, 0.4)
}

recommender = SimilarityMergedHybridRecommender(
    URM_train=URM_train,
    CFRecommender=p3alpha_recommender,
    CBFRecommender=pure_SVD_recommender,
    verbose=False
)


def BO_func(topK, alpha):
    recommender.fit(topK=int(topK), alpha=alpha)
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
p3alpha_recommender.load_model(folder_path='../../models/', file_name=p3alpha_recommender.RECOMMENDER_NAME)

pure_SVD_recommender = PureSVDItemRecommender(URM_train=URM_all)
pure_SVD_recommender.load_model(folder_path='../../models/', file_name=pure_SVD_recommender.RECOMMENDER_NAME)

recommender = SimilarityMergedHybridRecommender(
    URM_train=URM_all,
    CFRecommender=p3alpha_recommender,
    CBFRecommender=pure_SVD_recommender
)
recommender.fit(topK=int(hyperparameters['topK']), alpha=hyperparameters['alpha'])

recommender.save_model(folder_path='../../models/', file_name=recommender.RECOMMENDER_NAME)
