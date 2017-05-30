#%matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from numpy.testing import *
import scipy.optimize as op
from tifffile import TiffFile
import phase_ret3 as ph
import sys
import unittest
import nose

def imshow_grid(imgs, frac=1., clims=None):
    """A wrapper to conveniently display zoomed images in a grid."""
    n = len(imgs)
    fig = plt.figure(figsize=(4*n, 3))
    img_grid = ImageGrid(fig, 111, nrows_ncols=(1, n), axes_pad=.75,
    cbar_mode='each', cbar_pad='5%')
    for i, (ax, cbar, img) in enumerate(zip(img_grid,
                                        img_grid.cbar_axes,
                                        imgs)):
        N = img.shape[0] / 2.
        M = img.shape[1] / 2.
        im = ax.imshow(img, extent=(-M, M, N, -N),
        interpolation='nearest')
        ax.set_xlim(-frac*M, frac*M)
        ax.set_ylim(-frac*N, frac*N)
        if (clims is not None) and (clims[i] is not None):
            im.set_clim(clims[i])
        cbar.colorbar(im)
    return fig

def imshow_cmplx(img, **kwargs):
    imshow_grid([np.abs(img), np.angle(img)/2./np.pi], **kwargs)


# %% Testing the circle function.
class TestCircle:

    def setup(self):
        self.circ = ph.circle(8, 8, 4, 64)

    def test_circle_center_val(self):
        assert_equal(self.circ[8,8], 1.)

    def test_circle_edge_vals(self):
        assert_equal(self.circ[8,4], .5)
        assert_equal(self.circ[8,12], .5)
        assert_equal(self.circ[4,8], .5)
        assert_equal(self.circ[12,8], .5)


# %% Testing the crop function.
class TestCrop:

    def setup(self):
        self.arr = np.random.rand(16, 16)
        self.x0, self.y0 = 6, 6
        self.s = 4

    def test_crop_aligned_even(self):
        assert_array_equal(self.arr[4:8,4:8],
                           ph.crop(self.arr, self.x0, self.y0, self.s))

    def test_crop_aligned_odd(self):
        assert_array_equal(self.arr[4:9,4:9],
                           ph.crop(self.arr, self.x0, self.y0, self.s+1))

    def test_crop_nonaligned_even(self):
        assert_array_equal(self.arr[4:8,4:8],
                           ph.crop(self.arr, self.x0+.25, self.y0+.25, self.s))
        assert_array_equal(self.arr[5:9,4:8],
                           ph.crop(self.arr, self.x0+.25, self.y0+.75, self.s))

    def test_crop_nonaligned_odd(self):
        assert_array_equal(self.arr[4:9,4:9],
                           ph.crop(self.arr, self.x0+.25, self.y0+.25, self.s+1))
        assert_array_equal(self.arr[5:10,4:9],
                           ph.crop(self.arr, self.x0+.25, self.y0+.75, self.s+1))

# %% Run the tests!
if __name__ == '__main__':
    #unittest.main(argv=['ignored', '-v'], exit=False)
    result = nose.run(argv=['ignored', '-v'])
