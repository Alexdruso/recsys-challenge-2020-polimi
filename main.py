if __name__ == '__main__':


    from src.Hybrid.SimilarityMergedHybridRecommender import SimilarityMergedHybridRecommender
    from src.GraphBased.P3alphaRecommender import P3alphaRecommender
    from src.GraphBased.RP3betaCBFRecommender import RP3betaCBFRecommender
    from src.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
    from src.MatrixFactorization.PureSVDRecommender import PureSVDItemRecommender
    from src.Hybrid.GeneralizedMergedHybridRecommender import GeneralizedMergedHybridRecommender
    from src.SLIM_ElasticNet.SLIMElasticNetRecommender import MultiThreadSLIM_ElasticNet
    from src.Implicit.FeatureCombinedImplicitALSRecommender import FeatureCombinedImplicitALSRecommender
    from src.Hybrid.GeneralizedSimilarityMergedHybridRecommender import GeneralizedSimilarityMergedHybridRecommender
    from src.Utils.ICM_preprocessing import *
    from src.Utils.load_URM import load_URM
    from src.Utils.load_ICM import load_ICM
    from src.Utils.confidence_scaling import *
    from src.Utils.write_submission import write_submission

    URM_all = load_URM("in/data_train.csv")
    ICM_all = load_ICM("in/data_ICM_title_abstract.csv")

    ICM_combined = combine(ICM=ICM_all, URM = URM_all)

    # p3alpha_recommender = P3alphaRecommender(
    #     URM_train=URM_all,
    #     verbose=False
    # )
    #
    # p3alpha_recommender.fit(
    #     topK=int(212.8832860130684),
    #     alpha=0.4729294763382114,
    #     implicit=True
    # )

    rp3betaCombined_recommender= RP3betaCBFRecommender(
        URM_train=URM_all,
        ICM_train=ICM_combined,
        verbose=False
    )

    rp3betaCombined_recommender.fit(
        topK=int(527.4),
        alpha=0.4158,
        beta=0.288 ,
        implicit=False
    )

    IALS_recommender= FeatureCombinedImplicitALSRecommender(
        URM_train=URM_all,
        ICM_train=ICM_all,
        verbose=True
    )

    IALS_recommender.fit(
        factors=int(250),
        regularization=0.01,
        use_gpu=False,
        iterations=43,
        num_threads=4,
        confidence_scaling=linear_scaling_confidence,
        **{
            'URM': {"alpha": 50},
            'ICM': {"alpha": 50}
        }
    )

    SLIM_recommender = MultiThreadSLIM_ElasticNet(
            URM_train=ICM_combined.T,
            verbose=False
        )

    SLIM_recommender.fit(
        alpha=0.000490,
        l1_ratio=0.02911,
        topK=int(178),
        workers=4
    )

    SLIM_recommender.URM_train = URM_all

    # rp3betaCBF_recommender= RP3betaCBFRecommender(
    #     URM_train=URM_all,
    #     ICM_train=ICM_all,
    #     verbose=False
    # )
    #
    # rp3betaCBF_recommender.fit(
    #     topK=int(117.1),
    #     alpha=0.9882,
    #     beta=0.7703,
    #     implicit=False
    # )

    # itemKNN_recommender= ItemKNNCFRecommender(
    #     URM_train=URM_all,
    #     verbose=False
    # )
    #
    # itemKNN_recommender.fit(
    #     topK=100,
    #     shrink=50
    # )
    #
    # pureSVD_recommender= PureSVDItemRecommender(
    #     URM_train=URM_all,
    #     verbose=False
    # )
    #
    #
    # pureSVD_recommender.fit(
    #     num_factors=772,
    #     topK= 599
    # )

    recommender = GeneralizedMergedHybridRecommender(
        URM_train=URM_all,
        recommenders=[
            IALS_recommender,
            rp3betaCombined_recommender,
            SLIM_recommender
        ],
        verbose=False
    )

    recommender.fit(
        alphas=[
            0.5515927662083718,
            (1 - 0.5515927662083718),
            0.33515078397927905
        ]
    )

    write_submission(recommender=recommender, target_users_path="in/data_target_users_test.csv",
                     out_path='out/{}_submission.csv'.format(recommender.RECOMMENDER_NAME))
