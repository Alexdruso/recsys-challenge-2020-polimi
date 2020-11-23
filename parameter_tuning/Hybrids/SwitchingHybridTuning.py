from scipy import sparse as sps

from src.Base.Evaluation.Evaluator import EvaluatorHoldout
from src.Utils.load_ICM import load_ICM
from src.Utils.load_URM import load_URM
from src.Utils.ICM_preprocessing import *

URM_all = load_URM("../../in/data_train.csv")
ICM_all = load_ICM("../../in/data_ICM_title_abstract.csv")
from src.Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample

URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)

evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10], verbose=False)

ICM_augmented = combine(ICM_all, URM_train)

profile_length = np.ediff1d(URM_train.indptr)
block_size = int(len(profile_length)*0.25)

start_worst = 0
end_worst = block_size
end_normal = 3*block_size
end_best = min(4 * block_size, len(profile_length))
sorted_users = np.argsort(profile_length)

users_in_worst = set(sorted_users[start_worst:end_worst])

users_in_normal = set(sorted_users[end_worst:end_normal])

users_in_best = set(sorted_users[end_normal:end_best])


from src.Hybrid.SimilarityMergedHybridRecommender import SimilarityMergedHybridRecommender
from src.GraphBased.P3alphaRecommender import P3alphaRecommender
from src.GraphBased.RP3betaCBFRecommender import RP3betaCBFRecommender
from src.Hybrid.SwitchingHybrid import SwitchingHybrid

from bayes_opt import BayesianOptimization

p3alpha_recommender = P3alphaRecommender(URM_train=URM_train, verbose=False)
p3alpha_recommender.fit(topK=228,alpha=0.512,implicit=True)

rp3betaCBF_recommender = RP3betaCBFRecommender(URM_train=URM_train, ICM_train=ICM_all, verbose=False)
rp3betaCBF_recommender.fit(topK=63,alpha=0.221,beta=0.341,implicit=False)

recommender_worst = SimilarityMergedHybridRecommender(URM_train=URM_train,CFRecommender=p3alpha_recommender,CBFRecommender=rp3betaCBF_recommender,verbose=False)

recommender_normal = SimilarityMergedHybridRecommender(URM_train=URM_train,CFRecommender=p3alpha_recommender,CBFRecommender=rp3betaCBF_recommender,verbose=False)

recommender_best = SimilarityMergedHybridRecommender(URM_train=URM_train,CFRecommender=p3alpha_recommender,CBFRecommender=rp3betaCBF_recommender,verbose=False)

tuning_params = {
    "alpha": (0.1,0.9),
    "beta":(0.1,0.9),
    "gamma":(0.1,0.9),
    "topKWorst25":(10,500),
    "topKNormal":(10,500),
    "topKBest25":(10,500)
}


def BO_func(
        alpha,
        beta,
        gamma,
        topKWorst25,
        topKNormal,
        topKBest25
):

    recommender_worst.fit(topK=int(topKWorst25), alpha=alpha)
    recommender_normal.fit(topK=int(topKNormal), alpha=beta)
    recommender_best.fit(topK=int(topKBest25), alpha=gamma)

    recommender= SwitchingHybrid(

        URM_train=URM_train,
        recommenders=
        [
            recommender_worst,
            recommender_normal,
            recommender_best
        ],
        users_categories=
        [
            users_in_worst,
            users_in_normal,
            users_in_best
        ],
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
    init_points=150,
    n_iter=60,
)

print(optimizer.max)
#recommender.save_model(folder_path='../models/', file_name=recommender.RECOMMENDER_NAME)
