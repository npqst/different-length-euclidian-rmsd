
import numpy as np
from scipy.interpolate import splprep, splev


class StructuralUpsamplingRMSD():
    def __init__(self):
        return

    def calc_upsampled_rmsd(self, x, y, upsample_to=20):

        upsampled_x = self._upsample(x, new_length=upsample_to)
        upsampled_y = self._upsample(y, new_length=upsample_to)

        rmsd = self._rmsd(upsampled_x, upsampled_y)

        return rmsd

    def _upsample(self, input_series, new_length=20):
        tck, _ = splprep(
                        input_series.T,
                        u=np.asarray([*range(len(input_series))]),
                        s=0)
        upsampled = splev(np.linspace(0, len(input_series)-1, new_length), tck)
        return np.asarray(upsampled).T

    @staticmethod
    def _rmsd(x, y):
        assert x.shape == y.shape, f"""Shape of x ({x.shape})
                                    and y ({y.shape}) must be
                                    identical for RMSD"""
        rmsd = np.sqrt(np.sum(np.sum((x-y)**2, axis=-1))/len(x))
        return rmsd
