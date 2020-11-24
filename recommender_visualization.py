from src.Hybrid.SimilarityMergedHybridRecommender import SimilarityMergedHybridRecommender
from src.GraphBased.P3alphaRecommender import P3alphaRecommender
from src.GraphBased.RP3betaCBFRecommender import RP3betaCBFRecommender
from src.Utils.ICM_preprocessing import *
from src.Utils.load_URM import load_URM
from src.Utils.load_ICM import load_ICM
from src.Utils.write_submission import write_submission
from src.Utils.visualization import plotNumberRatingsMAPChart

URM_all = load_URM("in/data_train.csv")
ICM_all = load_ICM("in/data_ICM_title_abstract.csv")

from src.Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample

URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)

ICM_all = combine(ICM=ICM_all, URM = URM_train)

#p3alpha_recommender = P3alphaRecommender(URM_train=URM_all)
#p3alpha_recommender.fit(topK=221,alpha=0.5017,implicit=True)

rp3betaCBF_recommender = RP3betaCBFRecommender(URM_train=URM_train, ICM_train=ICM_all)
rp3betaCBF_recommender.fit(topK=529, alpha=0.453, beta=0.2266, implicit=False)

#recommender = SimilarityMergedHybridRecommender(
#    URM_train=URM_all,
#    CFRecommender=p3alpha_recommender,
#    CBFRecommender=rp3betaCBF_recommender
#)
#recommender.fit(topK=355, alpha=0.2222)

plotNumberRatingsMAPChart(rp3betaCBF_recommender,URM_train,URM_validation,0.25)