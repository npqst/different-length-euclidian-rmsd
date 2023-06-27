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

    def test_dp_rmsd_equivalence_on_examples(self):
        from Bio.PDB import PDBParser

        p = PDBParser()
        pdb7650 = p.get_structure(
            "7650",
            """different-length-euclidian-rmsd/examples/
            1279054_1_Paired_All_7650.pdb""",
        )
        pdb7744 = p.get_structure(
            "7744",
            """different-length-euclidian-rmsd/examples/
            1279059_1_Paired_All_7744.pdb""",
        )
        pdb11883 = p.get_structure(
            "11883",
            """different-length-euclidian-rmsd/examples/
            1287167_1_Paired_All_11883.pdb""",
        )

        ca_7650 = np.asarray(
            [x.coord for x in pdb7650.get_atoms() if x.id == "CA"]
        )
        ca_7744 = np.asarray(
            [x.coord for x in pdb7744.get_atoms() if x.id == "CA"]
        )
        ca_11883 = np.asarray(
            [x.coord for x in pdb11883.get_atoms() if x.id == "CA"]
        )
        dtw = DynamicTimeWarpDistance()

        pairs = [(ca_7650, ca_7744), (ca_11883, ca_7744), (ca_7650, ca_11883)]

        for cood_1, cood_2 in pairs:
            dtw_dist, dtw_path = dtw.dtw_dist(
                cood_1, cood_2, get_alignment=True
            )
            dist_matrix = dtw._calc_dist_matrix(cood_1, cood_2)
            rmsd = np.sqrt(
                np.sum(np.sum((cood_1 - cood_2) ** 2, axis=-1)) / len(cood_1)
            )
            diag_dist_mat_sum = np.sqrt(
                np.sum(dist_matrix * np.eye(len(cood_1))) / len(cood_1)
            )

            self.assertAlmostEqual(rmsd, diag_dist_mat_sum)

            diagonal_path = [(i, i) for i in range(len(cood_1) - 1)]
            if dtw_path != diagonal_path:
                self.assertTrue(rmsd >= dtw_dist)
            else:
                self.assertAlmostEqual(rmsd, dtw_dist)
