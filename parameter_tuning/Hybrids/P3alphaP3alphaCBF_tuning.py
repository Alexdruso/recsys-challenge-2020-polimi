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
from src.GraphBased.P3alphaCBFRecommender import P3alphaCBFRecommender

from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

from bayes_opt import BayesianOptimization

p3alpha_recommender = P3alphaRecommender(URM_train=URM_train, verbose=False)

p3alphaCBF_recommender = P3alphaCBFRecommender(URM_train=URM_train, ICM_train=ICM_all, verbose=False)

tuning_params = {
    "cfTopK": (210, 230),
    "cfAlpha": (0.45, 0.52),
    "cbfTopK": (430, 440),
    "cbfAlpha": (0.30, 0.32),
    "hybridTopK": (300, 600),
    "hybridAlpha": (0.6, 0.9)
}


def BO_func(
        cfTopK,
        cfAlpha,
        cbfTopK,
        cbfAlpha,
        hybridTopK,
        hybridAlpha
):
    p3alpha_recommender.fit(
        topK=int(cfTopK),
        alpha=cfAlpha,
        implicit=True
    )

    p3alphaCBF_recommender.fit(
        topK=int(cbfTopK),
        alpha=cbfAlpha,
        implicit=False
    )

    recommender = SimilarityMergedHybridRecommender(
        URM_train=URM_train,
        CFRecommender=p3alpha_recommender,
        CBFRecommender=p3alphaCBF_recommender,
        verbose=False
    )

    recommender.fit(
        topK=int(hybridTopK),
        alpha=hybridAlpha
    )
    result_dict, _ = evaluator_validation.evaluateRecommender(recommender)

    return result_dict[10]["MAP"]


optimizer = BayesianOptimization(
    f=BO_func,
    pbounds=tuning_params,
    verbose=5,
    random_state=5,
)

optimizer.maximize(
    init_points=100,
    n_iter=40,
)

hyperparameters = optimizer.max['params']

p3alpha_recommender = P3alphaRecommender(URM_train=URM_all)

p3alpha_recommender.fit(
    topK=int(hyperparameters['cfTopK']),
    alpha=hyperparameters['cfAlpha'],
    implicit=True,
    normalize_similarity=True
)

p3alphaCBF_recommender = P3alphaCBFRecommender(URM_train=URM_all, ICM_train=ICM_all)

p3alphaCBF_recommender.fit(
    topK=int(hyperparameters['cbfTopK']),
    alpha=hyperparameters['cbfAlpha'],
    implicit=True,
    normalize_similarity=True
)

recommender = SimilarityMergedHybridRecommender(
    URM_train=URM_all,
    CFRecommender=p3alpha_recommender,
    CBFRecommender=p3alphaCBF_recommender
)
recommender.fit(
    topK=int(hyperparameters['hybridTopK']),
    alpha=hyperparameters['hybridAlpha']
)

#recommender.save_model(folder_path='../models/', file_name=recommender.RECOMMENDER_NAME)
