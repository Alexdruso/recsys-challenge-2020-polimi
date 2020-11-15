from src.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from src.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from src.KNN.ItemKNNSimilarityHybridRecommender import ItemKNNSimilarityHybridRecommender
import csv
import numpy as np

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

data = pd.read_csv("in/data_train.csv")

import scipy.sparse as sps

userList = data['row'].tolist()
itemList = data['col'].tolist()
ratingList = data['data'].tolist()

URM_all = sps.coo_matrix((ratingList, (userList, itemList)))
URM_all = URM_all.tocsr()

metadata = pd.read_csv("in/data_ICM_title_abstract.csv")

itemICMList = metadata['row'].tolist()
featureList = metadata['col'].tolist()
weightList = metadata['data'].tolist()

ICM_all = sps.coo_matrix((weightList, (itemICMList, featureList)))
ICM_all = ICM_all.tocsr()

Similarity_1 = ItemKNNCFRecommender(URM_train=URM_all)
Similarity_1.fit(topK=100, shrink=50)
Similarity_1 = Similarity_1.W_sparse
Similarity_2 = ItemKNNCBFRecommender(URM_train=URM_all, ICM_train=ICM_all)
Similarity_2.fit(topK=500, shrink=10)
Similarity_2 = Similarity_2.W_sparse


recommender = ItemKNNSimilarityHybridRecommender(URM_train=URM_all, Similarity_1=Similarity_1, Similarity_2=Similarity_2)

recommender.fit(topK=500, alpha=0.67)

targetUsers = pd.read_csv("in/data_target_users_test.csv")['user_id']

# topNRecommendations = recommender.recommend(targetUsers.to_numpy(), cutoff=10)

targetUsers = targetUsers.tolist()

with open('out/{}_submission.csv'.format(recommender.RECOMMENDER_NAME), 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['user_id', 'item_list'])

    for userID in targetUsers:
        writer.writerow([userID, str(np.array(recommender.recommend(userID, 10)))[1:-1]])

