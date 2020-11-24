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
from src.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender

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

item_knncbf_optimizer = BayesianOptimization(
    f=None,
    pbounds={
    "topK": (10, 500),
    "shrink": (0, 200)
    }
)


p3alpha_recommender = P3alphaRecommender(URM_train=URM_train)

item_knncbf_recommender = ItemKNNCBFRecommender(URM_train=URM_train, ICM_train=ICM_all)

# New optimizer is loaded with previously seen points
load_logs(p3alpha_optimizer, logs=["logs/"+p3alpha_recommender.RECOMMENDER_NAME+'_logs.json'])

print(p3alpha_optimizer.max)

hyperparameters = p3alpha_optimizer.max['params']

p3alpha_recommender.fit(topK=int(hyperparameters['topK']), alpha=hyperparameters['alpha'], implicit=True)

load_logs(item_knncbf_optimizer, logs=["logs/"+item_knncbf_recommender.RECOMMENDER_NAME+'_jaccard_none_logs.json'])

print(item_knncbf_optimizer.max)

hyperparameters =item_knncbf_optimizer.max['params']

item_knncbf_recommender.fit(
    topK=int(hyperparameters['topK']),
    shrink=hyperparameters['shrink'],
    similarity='jaccard',
    feature_weighting='none'
)


tuning_params = {
    "topK": (10, 500),
    "alpha": (0.3, 0.4)
}

recommender = SimilarityMergedHybridRecommender(
    URM_train=URM_train,
    CFRecommender=p3alpha_recommender,
    CBFRecommender=item_knncbf_recommender,
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

item_knncbf_recommender = ItemKNNCBFRecommender(URM_train=URM_all, ICM_train=ICM_all)
item_knncbf_recommender.load_model(folder_path='../../models/', file_name=item_knncbf_recommender.RECOMMENDER_NAME + '_jaccard_none')

recommender = SimilarityMergedHybridRecommender(
    URM_train=URM_all,
    CFRecommender=p3alpha_recommender,
    CBFRecommender=item_knncbf_recommender
)
recommender.fit(topK=int(hyperparameters['topK']), alpha=hyperparameters['alpha'])

recommender.save_model(folder_path='../../models/', file_name=recommender.RECOMMENDER_NAME)
