import implicit
from ..Base.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from src.Utils.ICM_preprocessing import *


class FeatureCombinedImplicitALSRecommender(BaseMatrixFactorizationRecommender):
    """ImplicitALSRecommender recommender"""

    RECOMMENDER_NAME = "FeatureCombinedImplicitALSRecommender"

    def __init__(self, URM_train, ICM_train, verbose=False):
        super().__init__(URM_train, verbose=verbose)
        self.ICM_train = ICM_train

    def fit(self,
            factors=100,
            regularization=0.01,
            use_native=True, use_cg=True, use_gpu=False,
            iterations=15,
            calculate_training_loss=False, num_threads=0,
            confidence_scaling=None,
            **confidence_args
            ):

        self.rec = implicit.als.AlternatingLeastSquares(factors=factors, regularization=regularization,
                                                        use_native=use_native, use_cg=use_cg, use_gpu=use_gpu,
                                                        iterations=iterations,
                                                        calculate_training_loss=calculate_training_loss,
                                                        num_threads=num_threads)

        self.rec.fit(
            combine(
                confidence_scaling(
                    self.ICM_train,
                    **confidence_args['ICM']
                ),
                confidence_scaling(
                    self.URM_train,
                    **confidence_args['URM']
                )
            ),
            show_progress=self.verbose)

        self.USER_factors = self.rec.user_factors
        self.ITEM_factors = self.rec.item_factors
