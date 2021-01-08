if __name__ == '__main__':

    from src.Base.Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
    from src.Utils.ICM_preprocessing import *
    from src.Utils.confidence_scaling import *
    from src.Utils.load_ICM import load_ICM
    from src.Utils.load_URM import load_URM

    URM_all = load_URM("../../../in/data_train.csv")
    ICM_all = load_ICM("../../../in/data_ICM_title_abstract.csv")
    from src.Data_manager.split_functions.split_train_validation_random_holdout import \
        split_train_in_two_percentage_global_sample

    URMs_train = []
    URMs_validation = []
    ignore_users_list = []

    import numpy as np

    for k in range(5):
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
    for URM in URMs_train:
        ICMs_combined.append(combine(ICM=ICM_all, URM=URM))

    from src.Hybrid.GeneralizedMergedHybridRecommender import GeneralizedMergedHybridRecommender
    from src.Implicit.FeatureCombinedImplicitALSRecommender import FeatureCombinedImplicitALSRecommender
    from src.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
    from src.GraphBased.RP3betaCBFRecommender import RP3betaCBFRecommender

    from bayes_opt import BayesianOptimization

    IALS_recommenders = []
    rp3betaCBF_recommenders = []
    PureSVD_recommenders = []

    for index in range(len(URMs_train)):
        IALS_recommenders.append(
            FeatureCombinedImplicitALSRecommender(
                URM_train=URMs_train[index],
                ICM_train=ICM_all,
                verbose=True
            )
        )
        IALS_recommenders[index].fit(
            factors=500,
            regularization=0.01,
            use_gpu=False,
            iterations=94,
            num_threads=6,
            confidence_scaling=linear_scaling_confidence,
            **{
                'URM': {"alpha": 50},
                'ICM': {"alpha": 50}
            }
        )

        rp3betaCBF_recommenders.append(
            RP3betaCBFRecommender(
                URM_train=URMs_train[index],
                ICM_train=ICMs_combined[index],
                verbose=False
            )
        )

        rp3betaCBF_recommenders[index].fit(
            topK=int(741.3),
            alpha=0.4812,
            beta=0.2927,
            implicit=False
        )

        PureSVD_recommenders.append(
            PureSVDRecommender(
                URM_train=ICMs_combined[index].T,
                verbose=False
            )
        )

        PureSVD_recommenders[index].fit(
            num_factors=1000,
            random_seed=1,
            n_iter=1
        )

        PureSVD_recommenders[index].URM_train = URMs_train[index]

    tuning_params = {
        "alpha": (0, 1),
    }

    results = []


    def BO_func(
            alpha
    ):
        recommenders = []

        for index in range(len(URMs_train)):

            recommender = GeneralizedMergedHybridRecommender(
                URM_train=URMs_train[index],
                recommenders=[
                    IALS_recommenders[index],
                    rp3betaCBF_recommenders[index],
                    PureSVD_recommenders[index]
                ],
                verbose=False
            )

            recommender.fit(
                alphas=[
                    alpha*0.6354,
                    alpha*(1-0.6354),
                    1-alpha
                ]
            )

            recommenders.append(recommender)

        result = evaluator_validation.evaluateRecommender(recommenders)
        results.append(result)
        return sum(result) / len(result)


    optimizer = BayesianOptimization(
        f=BO_func,
        pbounds=tuning_params,
        verbose=5,
        random_state=5,
    )

    optimizer.maximize(
        init_points=90,
        n_iter=90,
    )

    recommender = GeneralizedMergedHybridRecommender(
                URM_train=URMs_train[0],
                recommenders=[
                    IALS_recommenders[0],
                    rp3betaCBF_recommenders[0],
                    PureSVD_recommenders[0]
                ],
                verbose=False
            )
    recommender.fit()

    import json

    with open("logs/FeatureCombined" + recommender.RECOMMENDER_NAME + "_best25_logs.json", 'w') as json_file:
        json.dump(optimizer.max, json_file)
