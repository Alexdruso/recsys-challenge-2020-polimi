from src.Base.Evaluation.Evaluator import EvaluatorHoldout
from src.MatrixFactorization.PureSVDRecommender import PureSVDItemRecommender
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
    "num_factors": (10, 500)
}

recommender = PureSVDItemRecommender(URM_train=URM_train, verbose=False)


def BO_func(topK, num_factors):
    recommender.fit(topK=int(topK), num_factors=int(num_factors))
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

logger = JSONLogger(path="logs/" + recommender.RECOMMENDER_NAME + "_logs.json")
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

optimizer.maximize(
    init_points=5,
    n_iter=10,
)

hyperparameters = optimizer.max['params']

recommender = PureSVDItemRecommender(URM_train=URM_all, verbose=False)
recommender.fit(topK=int(hyperparameters['topK']), num_factors=int(hyperparameters['num_factors']))

recommender.save_model(folder_path='../../models/', file_name=recommender.RECOMMENDER_NAME)
