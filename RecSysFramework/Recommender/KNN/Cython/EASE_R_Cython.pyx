#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: nonecheck=False
#cython: cdivision=True
#cython: unpack_method_calls=True
#cython: overflowcheck=False

#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

import cython
import tqdm

from cython.parallel import prange

import numpy as np
cimport numpy as np




cdef class EASE_RHelper:

    cdef int n_users, n_items

    cdef int[:] a_indptr, b_indptr
    cdef int[:] a_indices, b_indices
    cdef float[:] a_data, b_data

    def __init__(self, urm):

        self.n_users, self.n_items = urm.shape

        a = urm.transpose().tocsr(copy=True)
        b = urm.tocsc(copy=True)

        a.sort_indices()
        b.sort_indices()

        self.a_indptr = a.indptr
        self.a_indices = a.indices
        self.a_data = a.data.astype(np.float32)

        self.b_indptr = b.indptr
        self.b_indices = b.indices
        self.b_data = b.data.astype(np.float32)


    cpdef np.ndarray[np.float32_t, ndim=2] dot_product(self):

        cdef int[:] a_index, b_index
        cdef float[:, :] result_view

        cdef long i, j
        cdef int n_rows, n_cols

        n_rows = self.n_items
        n_cols = self.n_items

        a_index = np.zeros(n_rows, dtype=np.int32)
        b_index = np.zeros(n_rows, dtype=np.int32)
        
        result = np.zeros((self.n_items, self.n_items), dtype=np.float32)
        result_view = result

        progress = tqdm.tqdm(total=n_rows)
        progress.desc = 'Computing dot product'
        progress.refresh()

        for i in prange(n_rows, nogil=True, schedule='static'):
            for j in range(n_cols):
                a_index[i] = self.a_indptr[i]
                b_index[i] = self.b_indptr[j]
                while a_index[i] < self.a_indptr[i+1] and b_index[i] < self.b_indptr[j+1]:
                    if self.a_indices[a_index[i]] == self.b_indices[b_index[i]]:
                        result_view[i, j] += self.a_data[a_index[i]] * self.b_data[b_index[i]]
                        a_index[i] += 1
                        b_index[i] += 1
                    elif self.a_indices[a_index[i]] < self.b_indices[b_index[i]]:
                        a_index[i] += 1
                    else:
                        b_index[i] += 1
            with gil:
                progress.n += 1
                progress.refresh()

        return result
