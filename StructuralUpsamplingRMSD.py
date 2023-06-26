import numpy as np
from scipy.interpolate import splprep, splev


class StructuralUpsamplingRMSD:
    def __init__(self, upsampling_length=None):
        self.upsampling_length = upsampling_length
        return

    def calc_rmsd(self, x, y, new_length=None):
        if new_length is None:
            if self.upsampling_length is None:
                new_length = 20
            elif isinstance(self.upsampling_length, int):
                new_length = self.upsampling_length
        if max(len(x), len(y)) > new_length:
            new_length = max(len(x), len(y))
        return self.calc_upsampled_rmsd(x, y, upsample_to=new_length)

    def calc_upsampled_rmsd(self, x, y, upsample_to=20):
        upsampled_x = self._upsample(x, new_length=upsample_to)
        upsampled_y = self._upsample(y, new_length=upsample_to)

        rmsd = self._rmsd(upsampled_x, upsampled_y)

        return rmsd

    def _upsample(self, input_series, new_length=20):
        tck, _ = splprep(
            input_series.T, u=np.asarray([*range(len(input_series))]), s=0
        )
        upsampled = splev(
            np.linspace(0, len(input_series) - 1, new_length), tck
        )
        return np.asarray(upsampled).T

    @staticmethod
    def _rmsd(x, y):
        assert (
            x.shape == y.shape
        ), f"""Shape of x ({x.shape})
                                    and y ({y.shape}) must be
                                    identical for RMSD"""
        rmsd = np.sqrt(np.sum(np.sum((x - y) ** 2, axis=-1)) / len(x))
        return rmsd
