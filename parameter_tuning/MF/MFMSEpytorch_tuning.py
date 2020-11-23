from src.Base.Evaluation.Evaluator import EvaluatorHoldout
from src.MatrixFactorization.PyTorch.MF_MSE_PyTorch import MF_MSE_PyTorch
from src.Utils.load_ICM import load_ICM
from src.Utils.load_URM import load_URM

URM_all = load_URM("../../in/data_train.csv")
ICM_all = load_ICM("../../in/data_ICM_title_abstract.csv")
from src.Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample

URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)

evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10], verbose=False)

tuning_params = {
    "num_factors": (50, 500),
    "learning_rate": (1e-4, 1e-3)
}

recommender = MF_MSE_PyTorch(URM_train=URM_train)


def BO_func(
        num_factors,
        learning_rate
):
    recommender.fit(
        epochs=100,
        num_factors=int(num_factors),
        learning_rate=learning_rate,
        batch_size=1000,
        **{
            'epochs_min' : 10,
            'evaluator_object' : evaluator_validation,
            'stop_on_validation' : True,
            'validation_every_n' : 10,
            'validation_metric' : 'MAP',
            'lower_validations_allowed' : 3
        }
    )
    result_dict, _ = evaluator_validation.evaluateRecommender(recommender)

    return result_dict[10]["MAP"]


from bayes_opt import BayesianOptimization

optimizer = BayesianOptimization(
    f=BO_func,
    pbounds=tuning_params,
    verbose=5,
    random_state=5,
)

optimizer.maximize(
    init_points=15,
    n_iter=5
)

hyperparameters = optimizer.max['params']

recommender = MF_MSE_PyTorch(URM_train=URM_all)
recommender.fit(num_factors=int(hyperparameters['num_factors']),
                learning_rate=hyperparameters['learning_rate'])

#recommender.save_model(folder_path='../models/', file_name=recommender.RECOMMENDER_NAME)
