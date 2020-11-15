def load_URM(file_path):
    import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

    data = pd.read_csv(file_path)

    import scipy.sparse as sps

    user_list = data['row'].tolist()
    item_list = data['col'].tolist()
    rating_list = data['data'].tolist()

    return sps.coo_matrix((rating_list, (user_list, item_list))).tocsr()