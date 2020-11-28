from src.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from src.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from src.Utils.ICM_preprocessing import *
from src.Utils.load_ICM import load_ICM
from src.Utils.load_URM import load_URM
from src.Utils.visualization import plotNumberRatingsMAPChart

URM_all = load_URM("in/data_train.csv")
ICM_all = load_ICM("in/data_ICM_title_abstract.csv")

from src.Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample

URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)

ICM_all = combine(ICM=ICM_all, URM = URM_train)

#p3alpha_recommender = P3alphaRecommender(URM_train=URM_all)
#p3alpha_recommender.fit(topK=221,alpha=0.5017,implicit=True)

#rp3betaCBF_recommender = EASE_R_Recommender(URM_train=URM_train, sparse_threshold_quota=1.0)
#rp3betaCBF_recommender.fit(topK=56, l2_norm=41476.92126107723)

#recommender = SimilarityMergedHybridRecommender(
#    URM_train=URM_all,
#    CFRecommender=p3alpha_recommender,
#    CBFRecommender=rp3betaCBF_recommender
#)
#recommender.fit(topK=355, alpha=0.2222)

recommender = ItemKNNCBFRecommender(URM_train=URM_train, ICM_train=ICM_all)

recommender.fit(shrink=41.3, topK=919, similarity='cosine', feature_weighting='BM25')

plotNumberRatingsMAPChart(recommender,URM_train,URM_validation,0.25)