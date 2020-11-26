from src.Utils.ICM_preprocessing import *
from src.Utils.load_ICM import load_ICM
from src.Utils.load_URM import load_URM

URM_all = load_URM("in/data_train.csv")
ICM_all = load_ICM("in/data_ICM_title_abstract.csv")

profile_length = np.ediff1d(URM_all.indptr)
block_size = int(len(profile_length)*0.25)

start_worst = 0
end_worst = block_size
end_normal = 3*block_size
end_best = min(4 * block_size, len(profile_length))
sorted_users = np.argsort(profile_length)

users_in_worst = set(sorted_users[start_worst:end_worst])


users_in_normal = set(sorted_users[end_worst+1:end_normal])

users_in_best = set(sorted_users[end_normal+1:end_best])


from src.Hybrid.SimilarityMergedHybridRecommender import SimilarityMergedHybridRecommender
from src.GraphBased.P3alphaRecommender import P3alphaRecommender
from src.GraphBased.RP3betaCBFRecommender import RP3betaCBFRecommender
from src.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender

ICM_combined = combine(ICM=ICM_all, URM=URM_all)

p3alpha_recommender = P3alphaRecommender(URM_train=URM_all, verbose=False)
p3alpha_recommender.fit(topK=210,alpha=0.45,implicit=True)

rp3betaCBF_recommender = RP3betaCBFRecommender(URM_train=URM_all, ICM_train=ICM_combined, verbose=False)
rp3betaCBF_recommender.fit(topK=536,alpha=0.41753274557496695,beta=0.2344960487580402,implicit=False)

recommender_worst = SimilarityMergedHybridRecommender(URM_train=URM_all,CFRecommender=p3alpha_recommender,CBFRecommender=rp3betaCBF_recommender,verbose=False)
recommender_worst.fit(topK=481, alpha=0.1)

p3alpha_recommender = P3alphaRecommender(URM_train=URM_all, verbose=False)
p3alpha_recommender.fit(topK=229,alpha=0.5059173017438977,implicit=True)

rp3betaCBF_recommender = RP3betaCBFRecommender(URM_train=URM_all, ICM_train=ICM_combined, verbose=False)
rp3betaCBF_recommender.fit(topK=516,alpha=0.4227727007111746,beta=0.23482852065641355,implicit=False)

recommender_normal = SimilarityMergedHybridRecommender(URM_train=URM_all,CFRecommender=p3alpha_recommender,CBFRecommender=rp3betaCBF_recommender,verbose=False)
recommender_normal.fit(topK=456,alpha= 0.13589376902040495)

p3alpha_recommender = ItemKNNCBFRecommender(URM_train=URM_all, ICM_train=ICM_combined)
p3alpha_recommender.fit(shrink=135, topK=983,similarity='cosine', feature_weighting='BM25')

rp3betaCBF_recommender = RP3betaCBFRecommender(URM_train=URM_all, ICM_train=ICM_combined, verbose=False)
rp3betaCBF_recommender.fit(topK=577,alpha=0.448,beta=0.2612,implicit=False)

recommender_best = SimilarityMergedHybridRecommender(URM_train=URM_all,CFRecommender=p3alpha_recommender,CBFRecommender=rp3betaCBF_recommender,verbose=False)
recommender_best.fit(topK=872, alpha=0.2776,)

import pandas as pd
import csv
import numpy as np

targetUsers = pd.read_csv("in/data_target_users_test.csv")['user_id']

targetUsers = targetUsers.tolist()

with open('out/SwitchedHybrid_submission.csv','w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['user_id', 'item_list'])

    for userID in targetUsers:
        if userID in users_in_normal:
            writer.writerow([userID, str(np.array(recommender_normal.recommend(userID, 10)))[1:-1]])
        elif userID in users_in_best:
            writer.writerow([userID, str(np.array(recommender_best.recommend(userID, 10)))[1:-1]])
        else:
            writer.writerow([userID, str(np.array(recommender_worst.recommend(userID, 10)))[1:-1]])

