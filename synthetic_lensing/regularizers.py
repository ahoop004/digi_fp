import numpy as np

try:
    import scipy.sparse as sp

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def build_laplacian_regularizer(n_pix: int):
    N = n_pix * n_pix
    if _HAS_SCIPY:
        data = []
        rows = []
        cols = []

        def add(r, c, v):
            rows.append(r)
            cols.append(c)
            data.append(v)

        def idx(i: int, j: int) -> int:
            return i * n_pix + j

        for i in range(n_pix):
            for j in range(n_pix):
                k = idx(i, j)
                add(k, k, 4.0)
                if j - 1 >= 0:
                    add(k, idx(i, j - 1), -1.0)
                if j + 1 < n_pix:
                    add(k, idx(i, j + 1), -1.0)
                if i - 1 >= 0:
                    add(k, idx(i - 1, j), -1.0)
                if i + 1 < n_pix:
                    add(k, idx(i + 1, j), -1.0)
        R = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    else:
        R = np.zeros((N, N), dtype=float)

        def idx(i: int, j: int) -> int:
            return i * n_pix + j

        for i in range(n_pix):
            for j in range(n_pix):
                k = idx(i, j)
                R[k, k] = 4.0
                if j - 1 >= 0:
                    R[k, idx(i, j - 1)] = -1.0
                if j + 1 < n_pix:
                    R[k, idx(i, j + 1)] = -1.0
                if i - 1 >= 0:
                    R[k, idx(i - 1, j)] = -1.0
                if i + 1 < n_pix:
                    R[k, idx(i + 1, j)] = -1.0
    return R
