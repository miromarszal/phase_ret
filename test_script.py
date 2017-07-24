import numpy as np
from numpy.testing import *
import pandas as pd
import pandas.util.testing as pdt
import phase as ph
import nose


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


class TestCSF:

    def setup(self):
        self.y, self.x = np.indices((16, 16))

    def test_peak_scalar(self):
        assert_equal(ph.CSF(0, 0, 4), 1)

    def test_peak_array(self):
        assert_equal(ph.CSF(self.x-8, self.y-8, 4)[8, 8], 1)

    def test_minima_scalar(self):
        assert_almost_equal(ph.CSF(0, 4, 4), 0)
        assert_almost_equal(ph.CSF(0, -4, 4), 0)
        assert_almost_equal(ph.CSF(4, 0, 4), 0)
        assert_almost_equal(ph.CSF(-4, 0, 4), 0)

    def test_minima_array(self):
        assert_almost_equal(ph.CSF(self.x-8, self.y-8, 4)[8, 4], 0)
        assert_almost_equal(ph.CSF(self.x-8, self.y-8, 4)[8, 12], 0)
        assert_almost_equal(ph.CSF(self.x-8, self.y-8, 4)[4, 8], 0)
        assert_almost_equal(ph.CSF(self.x-8, self.y-8, 4)[12, 8], 0)


class TestTotalPower:

    def setup(self):
        N = 512
        self.bg = 1250.
        amp = 32000.
        s = 10.
        y, x = np.indices((N,N))
        dx, dy = 5.27, -3.78
        r2 = (x - (N / 2 - dx)) ** 2 + (y - (N / 2 - dy)) ** 2
        gauss = amp * np.exp(-r2 / 2 / s ** 2)
        self.totp = gauss.sum()
        img = self.bg +  gauss
        self.result = ph.total_power(img, N/2 - dx, N/2 - dy,
                                     r1=100, r2=200, r3=300, x=x, y=y)

    def test_background(self):
        assert_almost_equal(self.result[3], self.bg)

    def test_total_power(self):
        assert_almost_equal(self.result[0], self.totp)


class TestLocatePeak:

    def setup(self):
        Nx, Ny = 32, 64
        self.amp = 32000.
        s = 10.
        y, x = np.indices((Ny, Nx))
        self.res = 16
        max_delta = 3.
        self.x0 = Nx/2 + (np.random.rand() - .5) * max_delta
        self.y0 = Ny/2 + (np.random.rand() - .5) * max_delta
        r_sq = (x - self.x0) ** 2 + (y -self.y0) ** 2
        self.img = self.amp * np.exp(-r_sq / 2 / s ** 2)
        self.result = ph.locate_peak(self.img, self.res)

    def test_x(self):
        assert_array_less(np.abs(self.x0 - self.result[0]), 1./self.res)

    def test_y(self):
        assert_array_less(np.abs(self.y0 - self.result[1]), 1./self.res)

    def test_amp(self):
        assert_approx_equal(self.result[2], self.amp, significant=3)


class TestAnalyzePeaks:

    def setup(self):
        N = 512
        n = 3
        window = 32
        self.res = 16
        r1 = 100
        r2 = 200
        r3 = 300
        amp0 = 20000.
        bg0 = 1200.
        s0 = 10.
        max_delta_pos = 30.
        max_delta_amp = 10000.
        max_delta_bg = 500.
        max_delta_s = 5.
        y, x = np.indices((N, N))
        self.imgs = np.ones((n, N, N), dtype=np.float64)
        self.reference_df = pd.DataFrame(index=range(n),
                                         columns=['x0', 'y0', 'amp', 'bg',
                                                  'tot power', 'norm amp'],
                                         dtype=np.float64)
        for i in range(n):
            x0 = N/2 + (np.random.rand() - .5) * max_delta_pos
            y0 = N/2 + (np.random.rand() - .5) * max_delta_pos
            r_sq = (x - x0) ** 2 + (y -y0) ** 2
            amp = amp0 + (np.random.rand() - .5) * max_delta_amp
            s = s0 + (np.random.rand() - .5) * max_delta_s
            bg = bg0 + (np.random.rand() - .5) * max_delta_bg
            gauss = amp * np.exp(-r_sq / 2 / s ** 2)
            totp = gauss.sum()
            self.imgs[i] = gauss + bg
            self.reference_df.iloc[i] = x0, y0, amp, bg, totp, amp / totp
        self.result_df = ph.analyze_peaks(self.imgs, window, self.res,
                                          r1, r2, r3, print_output=False)[0]

    def test_pos(self):
        assert_array_less(np.abs(self.result_df.loc[:,'x0':'y0'] -
                                 self.reference_df.loc[:,'x0':'y0']).as_matrix(),
                                 1./self.res)

    def test_amp(self):
        pdt.assert_almost_equal(self.result_df['amp'], self.reference_df['amp'],
                                check_less_precise=3)

    def test_bg(self):
        pdt.assert_almost_equal(self.result_df['bg'], self.reference_df['bg'],
                                check_less_precise=3)

    def test_totp(self):
        pdt.assert_almost_equal(self.result_df['tot power'],
                                self.reference_df['tot power'],
                                check_less_precise=3)

    def test_norm(self):
        pdt.assert_almost_equal(self.result_df['norm amp'],
                                self.reference_df['norm amp'],
                                check_less_precise=3)


class TestErrfFFTW:

    def setup(self):
        if not ph.FFTW_LOADED:
            raise nose.SkipTest('FFTW not loaded.')
        F = np.random.rand(4, 256, 256)
        Z = np.random.rand(4)
        wl = np.random.rand()
        ph0 = np.random.rand(256, 256)
        errf_numpy = ph.Errf(Z[0], Z[1:], F[0], F[1:], wl)
        errf_fftw = ph.Errf_FFTW(Z[0], Z[1:], F[0], F[1:], wl)
        self.E_numpy, self.dE_numpy = errf_numpy(ph0)
        self.E_fftw, self.dE_fftw = errf_fftw(ph0)

    def test_E(self):
        assert_approx_equal(self.E_fftw, self.E_numpy)

    def test_dE(self):
        assert_allclose(self.dE_fftw, self.dE_numpy)


class TestErrfCUDA:

    def setup(self):
        if not ph.CUDA_LOADED:
            raise nose.SkipTest('CUDA not loaded.')
        F = np.random.rand(4, 256, 256)
        Z = np.random.rand(4)
        wl = np.random.rand()
        ph0 = np.random.rand(256, 256)
        errf_numpy = ph.Errf(Z[0], Z[1:], F[0], F[1:], wl)
        errf_cuda = ph.Errf_CUDA(Z[0], Z[1:], F[0], F[1:], wl)
        self.E_numpy, self.dE_numpy = errf_numpy(ph0)
        self.E_cuda, self.dE_cuda = errf_cuda(ph0)

    def test_E(self):
        assert_approx_equal(self.E_cuda, self.E_numpy)

    def test_dE(self):
        assert_allclose(self.dE_cuda, self.dE_numpy)


class TestTransformsFFTW:

    def setup(self):
        if not ph.FFTW_LOADED:
            raise nose.SkipTest('FFTW not loaded.')
        N = 256
        self.U = np.random.rand(N, N) + 1.j * np.random.rand(N, N)
        self.z = np.random.rand()
        self.wl = np.random.rand()
        self.trans_numpy = ph.Transforms(N)
        self.trans_fftw = ph.Transforms_FFTW(N)

    def test_fft(self):
        fft_numpy = self.trans_numpy.fft(self.U)

        fft_fftw = self.trans_fftw.fft(self.U)
        assert_allclose(fft_fftw, fft_numpy)

    def test_ifft(self):
        ifft_numpy = self.trans_numpy.ifft(self.U)
        ifft_fftw = self.trans_fftw.ifft(self.U)
        assert_allclose(ifft_fftw, ifft_numpy)

    def test_fraun_shift(self):
        fraun_numpy_pos = self.trans_numpy.fraun(self.U, self.z, self.wl, True)
        fraun_fftw_pos = self.trans_fftw.fraun(self.U, self.z, self.wl, True)
        fraun_numpy_neg = self.trans_numpy.fraun(self.U, -self.z, self.wl, True)
        fraun_fftw_neg = self.trans_fftw.fraun(self.U, -self.z, self.wl, True)
        assert_allclose(fraun_fftw_pos, fraun_numpy_pos)
        assert_allclose(fraun_fftw_neg, fraun_numpy_neg)

    def test_fraun_noshift(self):
        fraun_numpy_pos = self.trans_numpy.fraun(self.U, self.z, self.wl, False)
        fraun_fftw_pos = self.trans_fftw.fraun(self.U, self.z, self.wl, False)
        fraun_numpy_neg = self.trans_numpy.fraun(self.U, -self.z, self.wl, False)
        fraun_fftw_neg = self.trans_fftw.fraun(self.U, -self.z, self.wl, False)
        assert_allclose(fraun_fftw_pos, fraun_numpy_pos)
        assert_allclose(fraun_fftw_neg, fraun_numpy_neg)

    def test_asp(self):
        raise nose.SkipTest('To be fixed.')
        asp_numpy_pos = self.trans_numpy.asp(self.U, self.z, self.wl)
        asp_fftw_pos = self.trans_fftw.asp(self.U, self.z, self.wl)
        asp_numpy_neg = self.trans_numpy.asp(self.U, -self.z, self.wl)
        asp_fftw_neg = self.trans_fftw.asp(self.U, -self.z, self.wl)
        assert_allclose(asp_fftw_pos, asp_numpy_pos)
        assert_allclose(asp_fftw_neg, asp_numpy_neg)


class TestTransformsCUDA:

    def setup(self):
        if not ph.CUDA_LOADED:
            raise nose.SkipTest('CUDA not loaded.')
        N = 256
        self.U = np.random.rand(N, N) + 1.j * np.random.rand(N, N)
        self.z = np.random.rand()
        self.wl = np.random.rand()
        self.trans_numpy = ph.Transforms(N)
        self.trans_cuda = ph.Transforms_CUDA(N)

    def test_fft(self):
        fft_numpy = self.trans_numpy.fft(self.U)
        fft_cuda = self.trans_cuda.fft(self.U)
        assert_allclose(fft_cuda, fft_numpy)

    def test_ifft(self):
        ifft_numpy = self.trans_numpy.ifft(self.U)
        ifft_cuda = self.trans_cuda.ifft(self.U)
        assert_allclose(ifft_cuda, ifft_numpy)

    def test_fraun_shift(self):
        fraun_numpy_pos = self.trans_numpy.fraun(self.U, self.z, self.wl, True)
        fraun_cuda_pos = self.trans_cuda.fraun(self.U, self.z, self.wl, True)
        fraun_numpy_neg = self.trans_numpy.fraun(self.U, -self.z, self.wl, True)
        fraun_cuda_neg = self.trans_cuda.fraun(self.U, -self.z, self.wl, True)
        assert_allclose(fraun_cuda_pos, fraun_numpy_pos)
        assert_allclose(fraun_cuda_neg, fraun_numpy_neg)

    def test_fraun_noshift(self):
        fraun_numpy_pos = self.trans_numpy.fraun(self.U, self.z, self.wl, False)
        fraun_cuda_pos = self.trans_cuda.fraun(self.U, self.z, self.wl, False)
        fraun_numpy_neg = self.trans_numpy.fraun(self.U, -self.z, self.wl, False)
        fraun_cuda_neg = self.trans_cuda.fraun(self.U, -self.z, self.wl, False)
        assert_allclose(fraun_cuda_pos, fraun_numpy_pos)
        assert_allclose(fraun_cuda_neg, fraun_numpy_neg)

    def test_asp(self):
        asp_numpy_pos = self.trans_numpy.asp(self.U, self.z, self.wl)
        asp_cuda_pos = self.trans_cuda.asp(self.U, self.z, self.wl)
        asp_numpy_neg = self.trans_numpy.asp(self.U, -self.z, self.wl)
        asp_cuda_neg = self.trans_cuda.asp(self.U, -self.z, self.wl)
        assert_allclose(asp_cuda_pos, asp_numpy_pos)
        assert_allclose(asp_cuda_neg, asp_numpy_neg)


class TestZernike:

    def setup(self):
        N = 512
        a = 150.4
        v, u = np.indices((N, N))
        r = np.sqrt((u - N / 2.) ** 2 + (v - N / 2.) ** 2) / a
        p = np.arctan2(v - N / 2., u - N / 2.)
        self.R = ph.circle(N / 2., N / 2., a, N)
        self.zern = ph.Zernike(N / 2., N / 2., a, N, 8)
        self.Z = np.ones((8, N, N), dtype=np.float64)
        self.Z[1] = 2. * r * np.cos(p)
        self.Z[2] = 2. * r * np.sin(p)
        self.Z[3] = np.sqrt(3.) * (2. * r ** 2 - 1.)
        self.Z[4] = np.sqrt(6.) * r ** 2 * np.sin(2. * p)
        self.Z[5] = np.sqrt(6.) * r ** 2 * np.cos(2. * p)
        self.Z[6] = np.sqrt(8.) * (3. * r ** 3 - 2. * r) * np.sin(p)
        self.Z[7] = np.sqrt(8.) * (3. * r ** 3 - 2. * r) * np.cos(p)
        self.C = np.random.rand(8) * .1
        self.W = np.tensordot(self.C, self.Z, axes=(0,0))

    def test_poly(self):
        assert_allclose(self.zern.Z, self.Z)

    def test_poly_checksum(self):
        assert_allclose(self.zern.Z.sum(axis=(-2,-1)), self.Z.sum(axis=(-2,-1)))

    def test_wf(self):
        assert_allclose(self.zern(self.C), self.W)

    def test_fit(self):
        assert_allclose(self.zern.fit(self.W, 8), self.C, atol=1e-4)


# Run the tests!
if __name__ == '__main__':
    result = nose.run(argv=['ignored', '-v'])
