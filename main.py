from src.Hybrid.SimilarityMergedHybridRecommender import SimilarityMergedHybridRecommender
from src.GraphBased.P3alphaRecommender import P3alphaRecommender
from src.GraphBased.RP3betaCBFRecommender import RP3betaCBFRecommender
from src.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from src.MatrixFactorization.PureSVDRecommender import PureSVDItemRecommender
from src.Hybrid.GeneralizedSimilarityMergedHybridRecommender import GeneralizedSimilarityMergedHybridRecommender
from src.Utils.ICM_preprocessing import *
from src.Utils.load_URM import load_URM
from src.Utils.load_ICM import load_ICM
from src.Utils.write_submission import write_submission

URM_all = load_URM("in/data_train.csv")
ICM_all = load_ICM("in/data_ICM_title_abstract.csv")

ICM_combined = combine(ICM=ICM_all, URM = URM_all)

p3alpha_recommender = P3alphaRecommender(
    URM_train=URM_all,
    verbose=False
)

p3alpha_recommender.fit(
    topK=int(212.8832860130684),
    alpha=0.4729294763382114,
    implicit=True
)

rp3betaCombined_recommender= RP3betaCBFRecommender(
    URM_train=URM_all,
    ICM_train=ICM_combined,
    verbose=False
)

rp3betaCombined_recommender.fit(
    topK=int(525.3588205773788),
    alpha=0.42658191175355076,
    beta=0.2284685880641364,
    implicit=False
)

rp3betaCBF_recommender= RP3betaCBFRecommender(
    URM_train=URM_all,
    ICM_train=ICM_all,
    verbose=False
)

rp3betaCBF_recommender.fit(
    topK=int(58.16144182493173),
    alpha=0.26520214286626453,
    beta=0.36456352256640157,
    implicit=False
)

itemKNN_recommender= ItemKNNCFRecommender(
    URM_train=URM_all,
    verbose=False
)

itemKNN_recommender.fit(
    topK=100,
    shrink=50
)

pureSVD_recommender= PureSVDItemRecommender(
    URM_train=URM_all,
    verbose=False
)


pureSVD_recommender.fit(
    num_factors=772,
    topK= 599
)

recommender = GeneralizedSimilarityMergedHybridRecommender(
    URM_train=URM_all,
    similarityRecommenders=[
        p3alpha_recommender,
        rp3betaCombined_recommender,
        rp3betaCBF_recommender,
        itemKNN_recommender,
        pureSVD_recommender
    ],
        verbose=False
)

recommender.fit(
    topKs=[
        int(725.2251651408158),
        int(497.73913463467255),
        int(501.3131158958155),
        int(479.46922161562503)
    ],
    alphas=[
        0.5479200839179512,
        0.6137453754315466,
        0.7544056871377216,
        0.6162225526206547
    ]
)


write_submission(recommender=recommender, target_users_path="in/data_target_users_test.csv",
                 out_path='out/{}_submission.csv'.format(recommender.RECOMMENDER_NAME))
