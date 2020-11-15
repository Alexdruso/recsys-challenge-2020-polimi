from src.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from src.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from src.KNN.ItemKNNSimilarityHybridRecommender import ItemKNNSimilarityHybridRecommender
from src.KNN.ItemKNNCBFCFSimilarityHybridRecommender import ItemKNNCBFCFSimilarityHybridRecommender
from src.Utils.load_URM import load_URM
from src.Utils.load_ICM import load_ICM
from src.Utils.write_submission import write_submission
URM_all = load_URM("in/data_train.csv")

ICM_all = load_ICM("in/data_ICM_title_abstract.csv")

recommender = ItemKNNCBFCFSimilarityHybridRecommender(
    URM_train=URM_all,
    ICM_train=ICM_all,
    topK_knncf=100,
    shrink_knncf=50,
    topK_knncbf=500,
    shrink_knncbf=10
)

recommender.fit(topK=500, alpha=0.67)

write_submission(recommender=recommender, target_users_path="in/data_target_users_test.csv",out_path='out/{}_submission.csv'.format(recommender.RECOMMENDER_NAME))