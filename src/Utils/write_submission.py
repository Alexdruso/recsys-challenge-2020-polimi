def write_submission(recommender, target_users_path,out_path):
    import pandas as pd
    import csv
    import numpy as np

    targetUsers = pd.read_csv(target_users_path)['user_id']

    # topNRecommendations = recommender.recommend(targetUsers.to_numpy(), cutoff=10)

    targetUsers = targetUsers.tolist()

    with open(out_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['user_id', 'item_list'])

        for userID in targetUsers:
            writer.writerow([userID, str(np.array(recommender.recommend(userID, 10)))[1:-1]])

