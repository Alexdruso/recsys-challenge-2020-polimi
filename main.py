from src.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from src.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
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

recommender = PureSVDRecommender(URM_all)

recommender.fit()

targetUsers = pd.read_csv("in/data_target_users_test.csv")['user_id']

topNRecommendations = recommender.recommend(targetUsers.to_numpy(), cutoff=10)

targetUsers = targetUsers.tolist()

with open('out/submission.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['user_id', 'item_list'])

    for userID in targetUsers:
        writer.writerow([userID, str(np.array(recommender.recommend(userID, 10)))[1:-1]])