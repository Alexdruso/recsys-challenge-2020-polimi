import numpy as np
from src.Base.Evaluation.Evaluator import EvaluatorHoldout

def plotNumberRatingsMAPChart(recommender, URM_train, URM_test, granularity, cutoff = 10):

    profile_length = np.ediff1d(URM_train.indptr)
    block_size = int(len(profile_length) * granularity)
    sorted_users = np.argsort(profile_length)

    num_blocks = int(len(profile_length)/block_size)

    MAP_per_group = []

    for group_id in range(0, num_blocks):
        start_pos = group_id * block_size
        end_pos = min((group_id + 1) * block_size, len(profile_length))

        users_in_group = sorted_users[start_pos:end_pos]

        users_in_group_p_len = profile_length[users_in_group]

        print("Group {}, average p.len {:.2f}, min {}, max {}".format(group_id,
                                                                      users_in_group_p_len.mean(),
                                                                      users_in_group_p_len.min(),
                                                                      users_in_group_p_len.max()))


    for group_id in range(0, num_blocks):
        start_pos = group_id * block_size
        end_pos = min((group_id + 1) * block_size, len(profile_length))

        users_in_group = sorted_users[start_pos:end_pos]

        users_in_group_p_len = profile_length[users_in_group]

        print("Group {}, average p.len {:.2f}, min {}, max {}".format(group_id,
                                                                          users_in_group_p_len.mean(),
                                                                          users_in_group_p_len.min(),
                                                                          users_in_group_p_len.max()))

        users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
        users_not_in_group = sorted_users[users_not_in_group_flag]

        evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[cutoff], ignore_users=users_not_in_group)

        results, _ = evaluator_test.evaluateRecommender(recommender)
        MAP_per_group.append(results[cutoff]["MAP"])

    import matplotlib.pyplot as pyplot

    pyplot.plot(MAP_per_group, label=recommender.RECOMMENDER_NAME)
    pyplot.ylabel('MAP')
    pyplot.xlabel('User Group')
    pyplot.legend()
    pyplot.show()






