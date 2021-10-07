import numpy as np
import scipy.sparse as sps


def invert_dictionary(id_to_index):

    index_to_id = {}

    for id in id_to_index.keys():
        index = id_to_index[id]
        index_to_id[index] = id

    return index_to_id


def estimate_sparse_size(num_rows, topK):
    """
    :param num_rows: rows or colum of square matrix
    :param topK: number of elements for each row
    :return: size in Byte
    """

    num_cells = num_rows*topK
    sparse_size = 4*num_cells*2 + 8*num_cells

    return sparse_size


def seconds_to_biggest_unit(time_in_seconds, data_array=None):

    conversion_factor = [
        ("sec", 60),
        ("min", 60),
        ("hour", 24),
        ("day", 365),
    ]

    terminate = False
    unit_index = 0

    new_time_value = time_in_seconds
    new_time_unit = "sec"

    while not terminate:

        next_time = new_time_value/conversion_factor[unit_index][1]

        if next_time >= 1.0:
            new_time_value = next_time

            if data_array is not None:
                data_array /= conversion_factor[unit_index][1]

            unit_index += 1
            new_time_unit = conversion_factor[unit_index][0]

        else:
            terminate = True

    if data_array is not None:
        return new_time_value, new_time_unit, data_array

    else:
        return new_time_value, new_time_unit


def check_matrix(X, format='csc', dtype=np.float32):
    """
    This function takes a matrix as input and transforms it into the specified format.
    The matrix in input can be either sparse or ndarray.
    If the matrix in input has already the desired format, it is returned as-is
    the dtype parameter is always applied and the default is np.float32
    :param X:
    :param format:
    :param dtype:
    :return:
    """

    if sps.issparse(X):
        if format == 'csc' and not isinstance(X, sps.csc_matrix):
            return X.tocsc().astype(dtype)
        elif format == 'csr' and not isinstance(X, sps.csr_matrix):
            return X.tocsr().astype(dtype)
        elif format == 'coo' and not isinstance(X, sps.coo_matrix):
            return X.tocoo().astype(dtype)
        elif format == 'dok' and not isinstance(X, sps.dok_matrix):
            return X.todok().astype(dtype)
        elif format == 'bsr' and not isinstance(X, sps.bsr_matrix):
            return X.tobsr().astype(dtype)
        elif format == 'dia' and not isinstance(X, sps.dia_matrix):
            return X.todia().astype(dtype)
        elif format == 'lil' and not isinstance(X, sps.lil_matrix):
            return X.tolil().astype(dtype)
        return X.astype(dtype)
    elif isinstance(X, np.ndarray):
        X = sps.csr_matrix(X, dtype=dtype)
        X.eliminate_zeros()
        X.sort_indices()
        return check_matrix(X, format=format, dtype=dtype)


def reshapeSparse(sparseMatrix, newShape):

    if sparseMatrix.shape[0] > newShape[0] or sparseMatrix.shape[1] > newShape[1]:
        ValueError("New shape cannot be smaller than SparseMatrix. SparseMatrix shape is: {}, newShape is {}".format(
            sparseMatrix.shape, newShape))


    sparseMatrix = sparseMatrix.tocoo()
    newMatrix = sps.csr_matrix((sparseMatrix.data, (sparseMatrix.row, sparseMatrix.col)), shape=newShape)

    return newMatrix


def urm_to_coordinate_list(urm, neg_factor=1., random_seed=42):

    urm = check_matrix(urm, 'coo')
    n_users, n_items = urm.shape

    if random_seed is not None:
        np.random.seed(random_seed)

    neg_datasize = int(urm.data.size * neg_factor)

    neg_users = np.random.choice(n_users, neg_datasize).astype(np.int32)
    neg_items = np.random.choice(n_items, neg_datasize).astype(np.int32)
    neg_data = np.ones(neg_datasize).astype(np.float32)

    neg_matrix = sps.csr_matrix((neg_data, (neg_users, neg_items)), shape=(n_users, n_items))
    neg_matrix -= urm.astype(np.bool).astype(np.float32)
    neg_matrix = neg_matrix.tocoo()

    mask = neg_matrix.data > 0

    dataset_row = np.concatenate((urm.row, neg_matrix.row[mask]), axis=None)
    dataset_col = np.concatenate((urm.col, neg_matrix.col[mask]), axis=None)
    dataset_data = np.concatenate((urm.data, np.zeros(mask.sum(), dtype=np.float32)), axis=None)

    return dataset_row, dataset_col, dataset_data

