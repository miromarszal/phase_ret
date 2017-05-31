import numpy as np
from numpy.testing import *
import phase_ret3 as ph
import nose


# %% Testing the circle function.
class TestCircle:

    def setup(self):
        self.circ = ph.circle(8, 8, 4, 64)

    def test_center_val(self):
        assert_equal(self.circ[8,8], 1.)

    def test_edge_vals(self):
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

    def test_aligned_even(self):
        assert_array_equal(self.arr[4:8,4:8],
                           ph.crop(self.arr, self.x0, self.y0, self.s))

    def test_aligned_odd(self):
        assert_array_equal(self.arr[4:9,4:9],
                           ph.crop(self.arr, self.x0, self.y0, self.s+1))

    def test_nonaligned_even(self):
        assert_array_equal(self.arr[4:8,4:8],
                           ph.crop(self.arr, self.x0+.25, self.y0+.25, self.s))
        assert_array_equal(self.arr[5:9,4:8],
                           ph.crop(self.arr, self.x0+.25, self.y0+.75, self.s))

    def test_nonaligned_odd(self):
        assert_array_equal(self.arr[4:9,4:9],
                           ph.crop(self.arr, self.x0+.25, self.y0+.25, self.s+1))
        assert_array_equal(self.arr[5:10,4:9],
                           ph.crop(self.arr, self.x0+.25, self.y0+.75, self.s+1))


# %% Testing the CSF function
class TestCSF:

    def setup(self):
        self.y, self.x = np.indices((16, 16))

    def test_peak_scalar(self):
        assert_equal(ph.CSF(0, 0, 4), 1)

    def test_peak_array(self):
        assert_equal(ph.CSF(self.x-8, self.y-8, 4)[8, 8], 1)

    def test_minima_scalar(self):
        assert_almost_equal(ph.CSF(0, 4, 4), 0, decimal=3)
        assert_almost_equal(ph.CSF(0, -4, 4), 0, decimal=3)
        assert_almost_equal(ph.CSF(4, 0, 4), 0, decimal=3)
        assert_almost_equal(ph.CSF(-4, 0, 4), 0, decimal=3)

    def test_minima_array(self):
        assert_almost_equal(ph.CSF(self.x-8, self.y-8, 4)[8, 4], 0, decimal=3)
        assert_almost_equal(ph.CSF(self.x-8, self.y-8, 4)[8, 12], 0, decimal=3)
        assert_almost_equal(ph.CSF(self.x-8, self.y-8, 4)[4, 8], 0, decimal=3)
        assert_almost_equal(ph.CSF(self.x-8, self.y-8, 4)[12, 8], 0, decimal=3)


# %% Run the tests!
if __name__ == '__main__':
    result = nose.run(argv=['ignored', '-v'])
