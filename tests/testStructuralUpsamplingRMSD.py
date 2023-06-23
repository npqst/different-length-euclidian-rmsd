import unittest

import numpy as np
import matplotlib.pyplot as plt

from StructuralUpsamplingRMSD import StructuralUpsamplingRMSD


class TestStructuralUpsamplingRMSD(unittest.TestCase):
    def test_rmsd(self):
        x = np.random.rand(12, 3)
        y = np.zeros_like(x)
        rmsd = np.sqrt(np.sum(np.sum(x**2, axis=-1)) / 12)

        upsampling_rmsd = StructuralUpsamplingRMSD()

        self.assertEqual(rmsd, upsampling_rmsd._rmsd(x, y))

    def test_upsample(self):
        x = np.linspace(1, 5, 10) + np.sin(np.linspace(1, 5, 10))
        y = np.linspace(1, 10, 10) + np.sin(2 * np.linspace(1, 10, 10))
        z = np.linspace(-2, 30, 10) + np.sin(1.5 * np.linspace(-2, 30, 10))

        cood = np.stack([x, y, z], axis=-1)

        upsampling_rmsd = StructuralUpsamplingRMSD()
        upsampled_cood = upsampling_rmsd._upsample(cood, new_length=20)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Plot the line
        ax.plot3D(cood.T[0], cood.T[1], cood.T[2], "b")
        ax.plot3D(
            upsampled_cood.T[0], upsampled_cood.T[1], upsampled_cood.T[2], "r"
        )

        # Set labels for the axes
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()
        return

    def test_calc_upsampled_rmsd(self):
        x = np.linspace(1, 5, 10) + np.sin(np.linspace(1, 5, 10))
        y = np.linspace(1, 10, 10) + np.sin(2 * np.linspace(1, 10, 10))
        z = np.linspace(-2, 30, 10) + np.sin(1.5 * np.linspace(-2, 30, 10))

        cood = np.stack([x, y, z], axis=-1)
        base_cood = np.random.rand(*cood.shape)

        upsampling_rmsd = StructuralUpsamplingRMSD()

        self.assertAlmostEqual(
            upsampling_rmsd._rmsd(cood, base_cood),
            upsampling_rmsd.calc_upsampled_rmsd(
                cood, base_cood, upsample_to=x.shape[-1]
            ),
            places=6,
        )

        self.assertTrue(
            np.abs(
                upsampling_rmsd.calc_upsampled_rmsd(
                    cood, base_cood, upsample_to=x.shape[-1]
                )
                - upsampling_rmsd.calc_upsampled_rmsd(
                    cood, base_cood, upsample_to=20
                )
            )
            < 0.5
        )
