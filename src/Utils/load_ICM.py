def load_ICM(file_path):
    import pandas as pd
    import scipy.sparse as sps

    metadata = pd.read_csv(file_path)

    item_icm_list = metadata['row'].tolist()
    feature_list = metadata['col'].tolist()
    weight_list = metadata['data'].tolist()

    return sps.coo_matrix((weight_list, (item_icm_list, feature_list)))