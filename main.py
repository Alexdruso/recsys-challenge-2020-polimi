from src.Hybrid.P3alphaCBFMergedHybrid import P3alphaCBFMergedHybridRecommender
from src.GraphBased.P3alphaRecommender import P3alphaRecommender
from src.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from src.Utils.load_URM import load_URM
from src.Utils.load_ICM import load_ICM
from src.Utils.write_submission import write_submission

URM_all = load_URM("in/data_train.csv")
ICM_all = load_ICM("in/data_ICM_title_abstract.csv")

p3alpha_recommender = P3alphaRecommender(URM_train=URM_all)
p3alpha_recommender.load_model(folder_path='models/', file_name=p3alpha_recommender.RECOMMENDER_NAME)

item_knncbf_recommender = ItemKNNCBFRecommender(URM_train=URM_all, ICM_train=ICM_all)
item_knncbf_recommender.load_model(folder_path='models/', file_name=item_knncbf_recommender.RECOMMENDER_NAME)

recommender = P3alphaCBFMergedHybridRecommender(
    URM_train=URM_all,
    p3alpha_recommender=p3alpha_recommender,
    item_knncbf_recommender=item_knncbf_recommender
)
recommender.load_model(folder_path='models/', file_name=recommender.RECOMMENDER_NAME)

recommender.fit(topK=500, alpha=0.429)

write_submission(recommender=recommender, target_users_path="in/data_target_users_test.csv",
                 out_path='out/{}_submission.csv'.format(recommender.RECOMMENDER_NAME))
