from src.Hybrid.SimilarityMergedHybridRecommender import SimilarityMergedHybridRecommender
from src.GraphBased.P3alphaRecommender import P3alphaRecommender
from src.GraphBased.P3alphaCBFRecommender import P3alphaCBFRecommender
from src.Hybrid.ColdUsersTopPop import ColdUsersTopPop
from src.Utils.load_URM import load_URM
from src.Utils.load_ICM import load_ICM
from src.Utils.write_submission import write_submission

URM_all = load_URM("in/data_train.csv")
ICM_all = load_ICM("in/data_ICM_title_abstract.csv")

p3alpha_recommender = P3alphaRecommender(URM_train=URM_all)
p3alpha_recommender.fit(topK=int(225.5),alpha=0.4748,implicit=True)

p3alphaCBF_recommender = P3alphaCBFRecommender(URM_train=URM_all, ICM_train=ICM_all)
p3alphaCBF_recommender.fit(topK=436,alpha=0.3118,implicit=False)

recommender = SimilarityMergedHybridRecommender(
    URM_train=URM_all,
    CFRecommender=p3alpha_recommender,
    CBFRecommender=p3alphaCBF_recommender
)
recommender.fit(topK=393, alpha=0.7814)

write_submission(recommender=recommender, target_users_path="in/data_target_users_test.csv",
                 out_path='out/{}_submission.csv'.format(recommender.RECOMMENDER_NAME))
