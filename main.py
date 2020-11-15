from src.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from src.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from src.KNN.ItemKNNSimilarityHybridRecommender import ItemKNNSimilarityHybridRecommender
from src.Utils.load_URM import load_URM
from src.Utils.load_ICM import load_ICM
from src.Utils.write_submission import write_submission
URM_all = load_URM("in/data_train.csv")

ICM_all = load_ICM("in/data_ICM_title_abstract.csv")

Similarity_1 = ItemKNNCFRecommender(URM_train=URM_all)
Similarity_1.fit(topK=100, shrink=50)
Similarity_1 = Similarity_1.W_sparse
Similarity_2 = ItemKNNCBFRecommender(URM_train=URM_all, ICM_train=ICM_all)
Similarity_2.fit(topK=500, shrink=10)
Similarity_2 = Similarity_2.W_sparse


recommender = ItemKNNSimilarityHybridRecommender(URM_train=URM_all, Similarity_1=Similarity_1, Similarity_2=Similarity_2)

recommender.fit(topK=500, alpha=0.67)

write_submission(recommender=recommender, target_users_path="in/data_target_users_test.csv",out_path='out/{}_submission.csv'.format(recommender.RECOMMENDER_NAME))