import unittest
import numpy as np

from DynamicTimeWarpDistance import DynamicTimeWarpDistance


class TestDynamicTimeWarpDistance(unittest.TestCase):
    def test_calc_dist_matrix(self):
        x = np.random.rand(10, 3)
        y = np.zeros((9, 3))

        dtw = DynamicTimeWarpDistance()

        dist = np.tile(np.linalg.norm(x, axis=-1) ** 2, (len(y), 1)).T

        a = dtw._calc_dist_matrix(x, y)

        np.testing.assert_array_almost_equal(a, dist, decimal=10)

    def test_dp_path_search(self):
        test_matrix = np.asarray(
            [[0, 2, 2, 4], [-1, 1, 6, -10], [-2, -2, 1, 0]]
        )
        min_dist = -6
        dtw = DynamicTimeWarpDistance()
        a = dtw._dp_path_search(test_matrix, normalised=False)
        self.assertEqual(min_dist, a)

        a, b = dtw._dp_path_search(
            test_matrix, get_alignment=True, normalised=False
        )
        self.assertEqual(min_dist, a)
        self.assertEqual(b, [(0, 0), (0, 1), (0, 2), (1, 3)])

    def test_dp_path_search_rmsd_equivalence(self):
        x = np.random.rand(10, 3)
        y = np.zeros((10, 3))

        dtw = DynamicTimeWarpDistance()

        rmsd = np.sqrt(np.sum(np.sum(x**2, axis=-1)) / len(x))
        a = dtw.dtw_dist(x, y)
        self.assertAlmostEqual(a, rmsd, places=10)

        x = np.asarray(
            [[*range(12)], [*range(12)], [*range(12)]]
        ).T + np.random.rand(12, 3)
        y = np.asarray(
            [[*range(12)], [*range(12)], [*range(12)]]
        ).T + np.random.rand(12, 3)

        rmsd = np.sqrt(np.sum(np.sum((x - y) ** 2, axis=-1)) / len(x))
        a, b = dtw.dtw_dist(x, y, get_alignment=True)
        # check diagonal path through matrix:
        for i in range(len(x) - 1):
            self.assertEqual(b[i], (i, i))
        self.assertAlmostEqual(a, rmsd, places=10)
