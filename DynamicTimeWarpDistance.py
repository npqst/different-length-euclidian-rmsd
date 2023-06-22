import numpy as np


class DynamicTimeWarpDistance():
    def __init__(self):
        return

    def dtw_dist(self, x, y, get_alignment=False, normalised=True, rmsd_equivalent=True):
        dist_mat = self._calc_dist_matrix(x, y)
        
        if rmsd_equivalent is True:
            min_dist = self._dp_path_search(
                                        dist_mat,
                                        get_alignment=get_alignment,
                                        normalised=True
                                        )
            if get_alignment:
                return np.sqrt(min_dist[0]), min_dist[1]
            return np.sqrt(min_dist)
        else:
            min_dist = self._dp_path_search(
                                    dist_mat,
                                    get_alignment=get_alignment,
                                    normalised=normalised
                                    )
        return min_dist

    @staticmethod
    def _dp_path_search(dist_matrix,
                        get_alignment=False,
                        normalised=True
                        ):
        dtw_mat = np.ones_like(dist_matrix, dtype=float) * np.inf
        dtw_mat[0, 0] = 0.
        if normalised is True:
            path_length_mat = np.zeros_like(dist_matrix)

        def select_min_cost_origin(m, n):
            if m > 0 and n > 0:
                if not get_alignment and not normalised:
                    return min([
                                dtw_mat[m-1, n-1],
                                dtw_mat[m-1, n],
                                dtw_mat[m, n-1],
                                    ]), (None, None)
                else:
                    indices = [(m-1, n-1), (m-1, n), (m, n-1)]
                    dtw_origins = np.asarray([dtw_mat[idx[0], idx[1]]
                                              for idx in indices])
                    argmin = dtw_origins.argmin()
                    return dtw_origins[argmin], indices[argmin]
            elif m == 0 and n > 0:
                return dtw_mat[m, n-1], (m, n-1)
            elif n == 0 and m > 0:
                return dtw_mat[m-1, n], (m-1, n)
            else:
                return 0., (None, None)

        paths = np.zeros_like(dist_matrix).tolist() if get_alignment else None

        for i in range(0, dtw_mat.shape[0]):
            for j in range(0, dtw_mat.shape[1]):
                cost = dist_matrix[i, j]
                min_origin, idx_origin = select_min_cost_origin(i, j)
                dtw_mat[i, j] = cost + min_origin
                if normalised is True:
                    if idx_origin == (None, None):
                        path_length_mat[i, j] = 1
                    else:
                        path_length_mat[i, j] = path_length_mat[
                                            idx_origin[0],
                                            idx_origin[1]
                                            ] + 1
                if get_alignment:
                    paths[i][j] = idx_origin
        if get_alignment:
            origin = paths[-1][-1]
            min_path = []
            while origin != (None, None):
                min_path.insert(0, origin)
                origin = paths[origin[0]][origin[1]]
            if normalised:
                return dtw_mat[-1, -1]/path_length_mat[-1, -1], min_path 
            else:
                return dtw_mat[-1, -1], min_path
        if normalised:
            return dtw_mat[-1, -1] / path_length_mat[-1, -1]
        else:
            return dtw_mat[-1, -1]

    @staticmethod
    def _calc_dist_matrix(x, y):
        assert x.shape[-1] == y.shape[-1], f"""Last dimension of coordinates
                                            must match. Check input shapes of
                                            x ({x.shape}) and y ({y.shape})
                                            """
        pass
        x_expanded = np.tile(np.expand_dims(x, 1), (1, len(y), 1))
        y_expanded = np.tile(np.expand_dims(y, 0), (len(x), 1, 1))

        dist_mat = np.sum((x_expanded - y_expanded)**2, axis=-1)
        return dist_mat
