from ..Base.BaseRecommender import BaseRecommender


class ImplicitBaseRecommender(BaseRecommender):

    def __init__(self, URM_train, verbose=True):
        super(ImplicitBaseRecommender, self).__init__(URM_train=URM_train)
        self.verbose = verbose

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_not_compute=None,
                  remove_top_pop_flag=False, remove_custom_items_flag=False, return_scores=False):
        # items_to_be_recommended = [x
        #                            for x in self.data.ids_item
        #                            if x not in self.data.urm_train[user_id_array].indices]
        list_tuples_item_score = self.rec.recommend(user_id_array, self.URM_train,
                                                    filter_already_liked_items=remove_seen_flag, N=cutoff,
                                                    filter_items=items_to_not_compute)

        if (return_scores):
            return list_tuples_item_score
        else:
            list_items = []
            for tuple in list_tuples_item_score:
                item = tuple[0]
                list_items.append(item)
            return list_items