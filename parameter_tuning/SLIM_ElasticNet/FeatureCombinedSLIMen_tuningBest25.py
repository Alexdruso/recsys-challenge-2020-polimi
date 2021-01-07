if __name__ == '__main__':
    from src.Base.Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
    from src.SLIM_ElasticNet.SLIMElasticNetRecommender import MultiThreadSLIM_ElasticNet
    from src.Utils.ICM_preprocessing import *
    from src.Utils.load_ICM import load_ICM
    from src.Utils.load_URM import load_URM

    URM_all = load_URM("../../in/data_train.csv")
    ICM_all = load_ICM("../../in/data_ICM_title_abstract.csv")
    from src.Data_manager.split_functions.split_train_validation_random_holdout import \
        split_train_in_two_percentage_global_sample

    URMs_train = []
    URMs_validation = []
    ignore_users_list = []

    import numpy as np

    for k in range(3):
        URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)
        URMs_train.append(URM_train)
        URMs_validation.append(URM_validation)

        profile_length = np.ediff1d(URM_train.indptr)
        block_size = int(len(profile_length) * 0.25)

        start_pos = 3 * block_size
        end_pos = min(4 * block_size, len(profile_length))
        sorted_users = np.argsort(profile_length)

        users_in_group = sorted_users[start_pos:end_pos]

        users_in_group_p_len = profile_length[users_in_group]
        sorted_users = np.argsort(profile_length)

        users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
        ignore_users_list.append(sorted_users[users_not_in_group_flag])

    evaluator_validation = K_Fold_Evaluator_MAP(URMs_validation, cutoff_list=[10], verbose=False,
                                                ignore_users_list=ignore_users_list)
    ICMs_combined = []
    recommenders = []

    for URM in URMs_train:
        ICM_combined = combine(ICM=ICM_all, URM=URM)

        recommender = MultiThreadSLIM_ElasticNet(
                URM_train=ICM_combined.T,
                verbose=False
            )
        recommender.fit(
            alpha=0.00026894910579512645,
            l1_ratio=0.08074126876487486,
            topK=URM.shape[1]-1,
            max_iter=100,
            workers=6
        )

        recommender.URM_train = URM

        recommenders.append(
            recommender
        )

        print("recommender trained")

    result = evaluator_validation.evaluateRecommender(recommenders)
    topMap = sum(result) / len(result)
    bestTopK = np.inf

    print("The current top MAP is {}".format(topMap))

    from src.Base.Recommender_utils import ratingMatrixTopK

    for topK in [5000, 4000, 3000, 2000, 1500, 1000, 800, 700, 600, 400]:
        for index in range(len(URMs_train)):
            recommenders[index].W_sparse = ratingMatrixTopK(
                recommenders[index].W_sparse,
                k=topK
            )
        result = evaluator_validation.evaluateRecommender(recommenders)
        map = sum(result) / len(result)

        print("With topK = {} map is {}".format(topK, map))

        if(map>topMap):
            print("New topMap found!")
            topMap = map
            bestTopK = topK


    import json

    with open("logs/FeatureCombined" + recommenders[0].RECOMMENDER_NAME + "_best25_logs.json", 'w') as json_file:
        json.dump({"target": topMap, "params": {"topK": bestTopK}}, json_file)
