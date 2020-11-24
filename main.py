from src.Hybrid.SimilarityMergedHybridRecommender import SimilarityMergedHybridRecommender
from src.GraphBased.P3alphaRecommender import P3alphaRecommender
from src.GraphBased.RP3betaCBFRecommender import RP3betaCBFRecommender
from src.Utils.ICM_preprocessing import *
from src.Utils.load_URM import load_URM
from src.Utils.load_ICM import load_ICM
from src.Utils.write_submission import write_submission

URM_all = load_URM("in/data_train.csv")
ICM_all = load_ICM("in/data_ICM_title_abstract.csv")

ICM_all = combine(ICM=ICM_all, URM = URM_all)

#p3alpha_recommender = P3alphaRecommender(URM_train=URM_all)
#p3alpha_recommender.fit(topK=221,alpha=0.5017,implicit=True)

rp3betaCBF_recommender = RP3betaCBFRecommender(URM_train=URM_all, ICM_train=ICM_all)
rp3betaCBF_recommender.fit(topK=529, alpha=0.453, beta=0.2266, implicit=False)

#recommender = SimilarityMergedHybridRecommender(
#    URM_train=URM_all,
#    CFRecommender=p3alpha_recommender,
#    CBFRecommender=rp3betaCBF_recommender
#)
#recommender.fit(topK=355, alpha=0.2222)

write_submission(recommender=rp3betaCBF_recommender, target_users_path="in/data_target_users_test.csv",
                 out_path='out/FeatureCombined{}_submission.csv'.format(rp3betaCBF_recommender.RECOMMENDER_NAME))
