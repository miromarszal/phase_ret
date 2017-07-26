# -*- coding: utf-8 -*-
"""phase.py

Utility functions and classes for phase retrieval, image analysis
and Zernike polynomial fitting.

author: Miroslaw Marszalek
"""

import numpy as np
from scipy.ndimage import extrema
from scipy.special import j1
from scipy.signal import resample
import scipy.fftpack as spfft
from pandas import DataFrame
import os
import pkg_resources
fac = np.math.factorial

# Optionally import pyfftw for faster FFTs.
try:
    import pyfftw
    FFTW_LOADED = True
except ImportError:
    FFTW_LOADED = False
    print('Failed to import pyfftw.')

# Optionally import pycuda and skcuda for accelerated calculations on GPU
# and compile the kernels.
try:
    import pycuda.autoinit
    import pycuda.driver as drv
    import pycuda.gpuarray as gpa
    from pycuda.compiler import SourceModule
    from skcuda import fft as skfft
    kernels_b = pkg_resources.resource_string(__name__, 'kernels.cu')
    kernels = SourceModule(kernels_b.decode())
    CUDA_LOADED = True
except ImportError:
    CUDA_LOADED = False
    print('Failed to import pycuda/skcuda.')
except FileNotFoundError:
    CUDA_LOADED = False
    print('Failed to load kernels.cu.')
except drv.CompileError as err:
    CUDA_LOADED = False
    print('Failed to compile kernels:')
    print(err.msg)
    print('command: ' + ' '.join(err.command_line))
    print(err.stdout)
    print(err.stderr)

# Other constants.
NB64 = np.nbytes[np.float64]
NUM_CPU = os.cpu_count()


def circle(x0, y0, r, N):
    """Creates a circle of specified radius and position.

    Uses area sampling assuming a pixel to be a circle of radius 1/2
    and taking the intersection area as the pixel value.  Pixels
    lying on the edge are thus weighted according to their overlap
    with the circle.  Coordinates mean pixel centers, so (0, 0)
    means the center of (0, 0) pixel and (0, .5) means the (0, 0)
    pixel's right edge center, which is consistent with the
    convention used by the imshow command.

    Args:
        x0, y0: Circle center coordinates.
        r: Circle radius.
        N: Width of the image to be returned.

    Returns:
        An NxN numpy.array with values of 1 inside the circle,
        0 outside of it, and weighted accordingly on the edge.
    """
    circ = np.zeros((N, N))
    y, x = np.indices((N, N))
    d = 2. * (np.sqrt((x - x0) ** 2. + (y - y0) ** 2.) - r)
    idx = (d < 1.) * (d > -1.)
    edge = d[idx]
    circ[d <= -1.] = 1.
    circ[idx] = (np.arccos(edge) - edge * np.sqrt(1. - edge ** 2.)) / np.pi
    return circ


def crop(img, x0, y0, s):
    """Crops an image to the specified size.

    Args:
        img: An image to be cropped.
        x0, y0: Window center.
        s: Window size.

    Returns:
        A cropped view of the image.
    """
    x0 = int(round(x0)) - s // 2
    y0 = int(round(y0)) - s // 2
    img2 = img[y0:y0+s, x0:x0+s]
    return img2


def CSF(x, y, r0):
    """Aberration-free coherent spread function.

    Args:
        x, y: Coordinates, can be 2D arrays.
        r0: The Airy disk radius.
    """
    a = 3.83170597 / r0
    r = np.sqrt(x ** 2 + y ** 2)
    # Handle zero division.
    if isinstance(r, np.ndarray):
        csf = np.ones_like(r)
        csf[r != 0] = 2. * j1(a * r[r != 0]) / a / r[r != 0]
    else:
        csf = 2. * j1(a * r) / a / r if r else 1.
    return csf


def total_power(img, x0, y0, r1=100, r2=250, r3=300, x=None, y=None):
    """Calculates the total power of an image in a circular region.

    The algorithm sums all the pixel values from the image inside a
    circle of radius r1 centred on (x0, y0) and subtracts background
    taken as the mean value of the annulus between r2 and r3 centred
    on (x0, y0).

    Args:
        img: An image to be analyzed (numpy.array).
        x0, y0: Peak coordinates.
        r1, r2, r3: Radii for sectioning the image.
        x, y: Pixel coordnate arrays, to speed up the calculation.

    Returns a dict of results:
        totp: The total power.
        var_totp: Variance of the total power value.
        Nsig: Number of signal pixels used in the calculation.
        bg: The background level.
        var_bg: Variance of the background level.
        Nbg: Number of background pixels used in the calculation.
    """
    if x is None or y is None:
        y, x = np.indices(img.shape)
    # Sectioning the image into the signal and background parts
    img_sig = img[(x - x0) ** 2 + (y - y0) ** 2 <= r1 ** 2]
    img_bg = img[((x - x0) ** 2 + (y - y0) ** 2 <= r3 ** 2)
                 * ((x - x0) ** 2 + (y - y0) ** 2 >= r2 ** 2)]
    Nsig = img_sig.size
    Nbg = img_bg.size
    # Estimating the background and the total power
    bg = img_bg.mean()
    var_bg = img_bg.var(ddof=1)
    totp = img_sig.sum() - Nsig * bg
    var_totp = totp + Nsig * (1. + 1. / Nbg) * var_bg
    return totp, var_totp, Nsig, bg, var_bg, Nbg


def locate_peak(img, res=16):
    """Locates the maximum in the image via Fourier resampling.

    Args:
        img: The image to be resampled.
        res: Resampling rate, a pixel is divided into res**2 px.

    Returns:
        x0, y0: Peak coordinates.
        amp: Peak amplitude.
    """
    Ny, Nx = img.shape
    y, x = np.arange(Ny), np.arange(Nx)
    # Resample.
    imgResY, yRes = resample(img, res * Ny, y, axis=0)
    imgRes, xRes = resample(imgResY, res * Nx, x, axis=1)
    # Get positions and amplitude.
    y0Res, x0Res = np.unravel_index(imgRes.argmax(), imgRes.shape)
    y0, x0 = yRes[y0Res], xRes[x0Res]
    return x0, y0, imgRes.max()


def analyze_peaks(stack, window=32, res=16, r1=100, r2=250, r3=300,
                 x0=None, y0=None, index=None, print_output=True):
    """Finds peaks in a Tiff stack and does several measurements.

    Accepts stack of images and performs image analysis on it,
    returning a set of measurements for each page in the stack.
    Peak location (x0, y0) is extracted with sub-pixel accuracy by
    resampling with the Fourier method in a given window around the
    maximum and reading out the maximum position.  Background and
    total power are estimated via the total_power function (see
    below).  The amplitude from the resampled image is then divided
    by the total power to yield the normalized amplitude.

    Args:
        stack: A 3D numpy.array representing the stack of images
               to be analyzed.
        window: Size of the window used to resample an image.
        res: Resampling rate, a pixel is divided into res^2 pixels.
        r1, r2, r3: Radii for total_power (see below).
        x0, y0: If both are not None, peak location will be skipped.
                Instead, the peak position will be fixed at (x0, y0).
        index: List of indices to be passed to DataFrames holding
               the results.  If None, a sequential index will be used.
        print_output: If true, measurement results will be printed.

    Returns:
        parameters, variances : pandas.DataFrame of measurement
            results, each row in a frame corresponds to a page of
            the stack.

    Columns correspond to:
    x0, y0     -- sub-pixel peak location
    amp        -- peak amplitude of the resampled image after
                  background subtraction
    bg         -- background level
    tot power  -- total power of the signal
    norm amp   -- peak amplitude of the resampled image normalized
                  to the total power
    """
    if index is None:
        index = np.arange(len(stack))
    parameters = DataFrame(columns=['x0', 'y0', 'amp', 'bg', 'tot power',
                                    'norm amp'], index=index, dtype='float64')
    variances = DataFrame(columns=['amp', 'bg', 'tot power', 'norm amp'],
                          index=index, dtype='float64')
    if x0 is None or y0 is None:
        locate = True
    else:
        locate = False

    # Loop over all images in the stack
    for i, img in enumerate(stack):
        if locate:
            y0Cp, x0Cp = extrema(img)[3]  # Coarse maximum location
            imgCp = crop(img, x0Cp, y0Cp, window)  # Windowing
            # Image resampling, getting the sub-pixel peak position
            x0, y0, amp = locate_peak(imgCp, res)
            # Converting peak coordinates from window to full image
            x0 += x0Cp - window / 2
            y0 += y0Cp - window / 2
        else:
            imgCp = crop(img, x0, y0, window)
            amp = img[y0, x0]
        # Measurements and normalization
        totp, var_totp, Nsig, bg, var_bg, Nbg = total_power(img, x0, y0,
                                                            r1, r2, r3)
        amp -= bg
        var_amp = amp + (1. + 1. / Nbg) * var_bg
        amp_norm = amp / totp
        var_amp_norm = amp_norm ** 2 * (var_amp / amp ** 2
                                        + var_totp / totp ** 2)
        # Saving to containers
        parameters.iloc[i] = x0, y0, amp, bg, totp, amp_norm
        variances.iloc[i] = var_amp, var_bg, var_totp, var_amp_norm

    if print_output:
        print('parameters:\n')
        print(parameters)
        print('\nvariances:\n')
        print(variances)
    return parameters, variances


class Errf(object):
    """Error function for phase retrieval via non-linear optimization.

    Defines a callable compatible with the scipy.optimize.minimize function,
    i.e. accepting a parameter vector (phase of a complex field) and returning
    the error metric and the gradient.  This is the core class using NumPy
    and SciPy for calculations.  Subclassed by Errf_FFTW and Errf_CUDA.

    Args:
        zj: Position of the master plane in image-space pixels.
        zk: ndarray of slave plane positions in image-space pixels.
        Fj: Amplitude distribution in the master plane.
        Fk: Amplitude distributions in slave planes.
        wl: Light wavelength in image-space pixels.

    Attributes:
        zj, zk, Fj, Fk: See above.
        N: Width of a single image, necessary to define transforms.
        num: Number of slave planes.
        Tk: Transfer array for angular spectrum propagation from the
            master plane to slave planes.
        Tkconj: Complex conjugate of Tk, defines inverse transforms.
    """

    def __init__(self, zj, zk, Fj, Fk, wl):
        self.zj = zj
        self.zk = zk
        self.Fj = Fj
        self.Fk = Fk
        self.N = self.Fj.shape[0]
        self.num = len(Fk)
        y, x = np.indices((self.N, self.N))
        x = spfft.fftshift(x - self.N/2)
        y = spfft.fftshift(y - self.N/2)
        self.Tk = np.exp(2.j * np.pi * (self.zk - self.zj)[:,None,None]
                             * np.sqrt(1. / wl ** 2 - (x ** 2 + y ** 2)
                                                    / self.N ** 2))
        self.Tkconj = self.Tk.conj()

    def __call__(self, ph):
        """Calculates the error metric for given phase estimate.

        Args:
            ph: Phase in the master plane.

        Returns:
            E: The error metric.
            dE: Gradient of the error metric.
        """
        Gj = self.Fj * np.exp(1.j * ph.reshape((self.N, self.N)))
        Gkj = spfft.ifft2(spfft.fft2(Gj) * self.Tk)
        E = np.sum((self.Fk - np.abs(Gkj)) ** 2)
        Gwjk = spfft.ifft2(spfft.fft2((self.Fk * Gkj / np.abs(Gkj) - Gkj))
                                                                  * self.Tkconj)
        dE = 2. * np.imag(Gj * np.sum(Gwjk.conj(), axis=0))
        return E, dE.ravel()


class Errf_FFTW(Errf):
    """Subclass of Errf utilizing pyfftw for faster transforms.

    Attributes:
        fft21: 2D FFT of a single array.
        fft2n: Batched 2D FFT of several arrays.
        ifft2n: Batched 2D IFFT.
    """

    def __init__(self, zj, zk, Fj, Fk, wl):
        Errf.__init__(self, zj, zk, Fj, Fk, wl)
        self.fft21 = pyfftw.builders.fft2(self.Fj.astype(np.complex128),
                                                        threads=NUM_CPU)
        self.fft2n = pyfftw.builders.fft2(self.Fk.astype(np.complex128),
                                                        threads=NUM_CPU)
        self.ifft2n = pyfftw.builders.ifft2(self.Fk.astype(np.complex128),
                                                          threads=NUM_CPU)

    def __call__(self, ph):
        Gj = self.Fj * np.exp(1.j * ph.reshape((self.N, self.N)))
        Gkj = self.ifft2n(self.fft21(Gj) * self.Tk)
        E = np.sum((self.Fk - np.abs(Gkj)) ** 2)
        Gwjk = self.ifft2n(self.fft2n((self.Fk * Gkj / np.abs(Gkj) - Gkj))
                                                                  * self.Tkconj)
        dE = 2. * np.imag(Gj * np.sum(Gwjk.conj(), axis=0))
        return E, dE.ravel()


class Errf_CUDA(Errf):
    """Subclass of Errf utilizing CUDA-accelerated computation.

    Uses pycuda and skcuda to facilitate integration with Python.
    Employs kernels defined in 'kernels.cu'.

    TO DO: Better (scalable) divisions into threadblocks.
    Currently threads are divided into N blocks x N threads per block,
    where N is the array width.  This makes for more readable code,
    but obviously puts a limit on scalability.
    """

    def __init__(self, zj, zk, Fj, Fk, wl):
        Errf.__init__(self, zj, zk, Fj, Fk, wl)

        ### Arrays to store intermediate results on the GPU.
        self.Fj = gpa.to_gpu(self.Fj)
        self.Fk = gpa.to_gpu(self.Fk)
        self.Tk = gpa.to_gpu(self.Tk)
        self.Tkconj = gpa.to_gpu(self.Tkconj)
        self.ph = gpa.empty_like(self.Fj)
        self.Gj = gpa.empty((self.N, self.N), np.complex128)
        self.Gkj = gpa.empty_like(self.Tk)
        self.Gwkj = gpa.empty_like(self.Tk)
        self.Gwjk = gpa.empty_like(self.Tk)
        self.Earr = gpa.empty_like(self.Fk)
        self.Ered = gpa.empty(self.N, np.float64)
        self.E = gpa.empty(1, np.float64)
        self.dE = gpa.empty_like(self.Fj)
        self.temp1 = gpa.empty_like(self.Gj)
        self.tempn = gpa.empty_like(self.Tk)

        ### FFT plans.
        self.fft21 = skfft.Plan(self.Gj.shape, self.Gj.dtype, self.Gj.dtype)
        self.fft2n = skfft.Plan(self.Gj.shape, self.Gj.dtype, self.Gj.dtype,
                                batch=self.num)

        ### Functions from 'kernels.cu'.
        self.get_Gj = kernels.get_function('get_Gj')
        self.mult_complex_1to1 = kernels.get_function('mult_complex_1to1')
        self.mult_complex_1ton = kernels.get_function('mult_complex_1ton')
        self.get_E_Gwkj = kernels.get_function('get_E_Gwkj')
        self.sum_E = kernels.get_function('sum_E')
        self.sum_Etot = kernels.get_function('sum_Etot')
        self.get_dE = kernels.get_function('get_dE')

    def __call__(self, ph):
        ### Preparing stuff.
        self.ph.set(ph.reshape((self.N, self.N)))
        self.get_Gj(self.Fj, self.ph, self.Gj, block=(self.N,1,1),
                    grid=(self.N,1))
        skfft.fft(self.Gj, self.temp1, self.fft21)
        self.mult_complex_1ton(self.temp1, self.Tk, self.tempn,
                               block=(self.N,1,1), grid=(self.N,self.num))
        skfft.ifft(self.tempn, self.Gkj, self.fft2n, scale=True)
        self.get_E_Gwkj(self.Fk, self.Gkj, self.Earr, self.Gwkj,
                        block=(self.N,1,1), grid=(self.N,self.num))

        ### Summing the error.
        self.sum_E(np.uint32(self.num), self.Earr, block=(self.N,1,1),
                   grid=(self.N,1))
        self.sum_Etot(self.Earr, self.Ered, block=(self.N//2,1,1),
                      grid=(self.N,1), shared=self.N//2*NB64)
        self.sum_Etot(self.Ered, self.E, block=(self.N//2,1,1),
                      shared=self.N//2*NB64)

        ### Calculating the gradient.
        skfft.fft(self.Gwkj, self.tempn, self.fft2n)
        self.mult_complex_1to1(self.tempn, self.Tkconj, self.Gwkj,
                               block=(self.N,1,1), grid=(self.N,self.num))
        skfft.ifft(self.Gwkj, self.Gwjk, self.fft2n, scale=True)
        self.get_dE(np.uint32(self.num), self.Gj, self.Gwjk, self.dE,
                    block=(self.N,1,1), grid=(self.N,1))

        return self.E.get()[0], self.dE.get().ravel()


class Transforms(object):
    """A container class for FFTs and diffraction integrals.

    Defines 2D FFTs for a particular size images being transformed and
    Fraunhofer diffraction and angular spectrum propagation integrals.
    Subclassed by Transforms_FFTW and Transforms_CUDA.

    Args:
        N: Image width to be used. Images must be rectangles.

    Attributes:
        N: See above.
        r2: Square of the real-space radial coordinate.
    """

    def __init__(self, N):
        self.N = N
        # Coordinate arrays
        y, x = np.indices((N,N))
        self.x = spfft.ifftshift(x).astype(float) - N/2
        self.y = spfft.ifftshift(y).astype(float) - N/2
        self.r2 = self.x ** 2 + self.y ** 2

    def fft(self, U):
        return spfft.fft2(U)

    def ifft(self, U):
        return spfft.ifft2(U)

    def fraun(self, U, z, wl):
        """Simulates light propagation according to the Fraunhofer integral.

        The length unit is the pixel size in the image space, that is
        the space of the output field for forward propagation (z>=0) and
        the space of the intput field for backward propagation (z<0).
        The fields are expected to be FFT-shifted before the operation.
        Performing the shift is left up to the user.

        Args:
            U: A complex NxN array representing the input field.
            z: Distance of propagation in image-space pixels.
            wl: Wavelength of light in image-space pixels.

        Returns:
            A complex NxN array representing the transformed field.
        """
        # Phase factors.
        ph1 = np.exp(2.j * np.pi * z / wl)
        ph2 = np.exp(1.j * np.pi / wl / z * self.r2)
        if z>=0:
            U2 = -1.j / self.N * ph1 * ph2 * self.fft(U)
        else:
            U2 = 1.j * self.N * ph1 * self.ifft(U * ph2)
        return U2

    def asp(self, U, z, wl):
        """Light propagation according to the angular spectrum propagation.

        In this case the pixel size is the same in both spaces and
        remains unchanged under propagation.

        Args:
            U: A complex NxN array representing the input field.
            z: Distance of propagation in pixels.
            wl: Wavelength of light in pixels.

        Returns:
            A complex NxN array representing the transformed field.
        """
        T = np.exp(2.j * np.pi * z * np.sqrt(1. / wl ** 2 - self.r2 / self.N ** 2))
        U2 = self.ifft(self.fft(U) * T)
        return U2


class Transforms_FFTW(Transforms):
    """Subclass of Transforms employing FFTW for faster computation.

    TO DO: The asp method consistently fails the tests. Probably
        related to the memory management of pyfftw. In normal usage
        everything seems to work fine.
    """

    def __init__(self, N):
        Transforms.__init__(self, N)
        A = pyfftw.empty_aligned((N,N), dtype=np.complex128)
        self.fftw = pyfftw.builders.fft2(A, threads=NUM_CPU)
        self.ifftw = pyfftw.builders.ifft2(A, threads=NUM_CPU)

    def fft(self, U):
        """Overrides Transforms.fft."""
        return self.fftw(U)

    def ifft(self, U):
        """Overrides Transforms.ifft."""
        return self.ifftw(U)


class Transforms_CUDA(Transforms):
    """Subclass of Transforms employing CUDA for faster computation.

    TO DO:
        * Errors accumulate somewhere. Every now and then tests will fail
            for the fraun function.
        * The same issue with threadblocks as in Errf_CUDA.
    """

    def __init__(self, N):
        Transforms.__init__(self, N)
        self.Uin = gpa.empty((self.N, self.N), np.complex128)
        self.Uout = gpa.empty((self.N, self.N), np.complex128)
        self.Utemp = gpa.empty((self.N, self.N), np.complex128)
        self.r2_shift = gpa.to_gpu(spfft.fftshift(self.r2))
        self.r2 = gpa.to_gpu(self.r2)
        self.fft_plan = skfft.Plan((self.N, self.N), np.complex128,
                                                     np.complex128)
        self.fftshift = kernels.get_function('fftshift')
        self.mult_T = kernels.get_function('mult_T')
        self.mult_ph12 = kernels.get_function('mult_ph12')
        self.mult_ph1 = kernels.get_function('mult_ph1')
        self.mult_ph2 = kernels.get_function('mult_ph2')

    def fft(self, U):
        """Overrides Transforms.fft."""
        self.Uin.set(U)
        skfft.fft(self.Uin, self.Uout, self.fft_plan)
        return self.Uout.get()

    def ifft(self, U):
        """Overrides Transforms.ifft."""
        self.Uin.set(U)
        skfft.ifft(self.Uin, self.Uout, self.fft_plan, scale=True)
        return self.Uout.get()

    def fraun(self, U, z, wl):
        """Overrides Transforms.fraun"""
        self.Uin.set(U)
        if z>=0:
            skfft.fft(self.Uin, self.Uout, self.fft_plan)
            self.mult_ph12(np.uint32(self.N), np.float64(z), np.float64(wl),
                           self.r2, self.Uout,
                           block=(self.N,1,1), grid=(self.N,1))
        else:
            self.mult_ph2(np.uint32(self.N), np.float64(z), np.float64(wl),
                          self.r2, self.Uin,
                          block=(self.N,1,1), grid=(self.N,1))
            skfft.ifft(self.Uin, self.Uout, self.fft_plan, scale=True)
            self.mult_ph1(np.uint32(self.N), np.float64(z), np.float64(wl),
                          self.Uout, block=(self.N,1,1), grid=(self.N,1))
        return self.Uout.get()

    def asp(self, U, z, wl):
        """Overrides Transforms.asp."""
        self.Uin.set(U)
        skfft.fft(self.Uin, self.Utemp, self.fft_plan)
        self.mult_T(np.uint32(self.N), np.float64(z), np.float64(wl),
                    self.r2, self.Utemp, block=(self.N,1,1), grid=(self.N,1))
        skfft.ifft(self.Utemp, self.Uout, self.fft_plan, scale=True)
        return self.Uout.get()


class Zernike(object):
    """A callable class for storing and fitting Zernike polynomials.

    Polynomials are indexed with a single index according to Noll's
    convention and normalized to their RMS values.

    Args:
        u0, v0: Coordinates of the pupil centre.
        a: Exit pupil aperture radius.
        N: Image size.
        jmax: Number of polynomials to be allocated.

    Attributes:
        a, N: See above.
        r: Normalized radial coordinate.
        p: Angular coordinate.
        R: A circle representing the pupil aperture.
        idx: (jmax)x2 array providing conversion between Noll's
            convention and the (n,m) indexing scheme.
        Z: (jmax)xNxN array representing Zernike polynomials.
    """

    def __init__(self, N, jmax):
        self.N = N
        self.jmax = jmax
        self.v, self.u = np.indices((self.N, self.N))

        # Allocating stuff.
        self.u0, self.v0 = 0., 0.
        self.a = 0.
        self.r = np.zeros((self.N, self.N), dtype=np.float64)
        self.p = np.zeros_like(self.r)
        self.R = np.zeros_like(self.r)
        self.Z = np.ones((jmax, N, N), dtype=np.float64)

    def get_indices(self, j):
        """Converts Noll's j index into (n, m) indices.

        Probably there is a smarter way to do that...
        """
        # n is determined from the sum of arithmetic series.
        n = int(np.ceil((np.sqrt(1. + 8. * j) - 1.) / 2.) - 1.)
        k = j - n * (n + 1) // 2 - 1  # Position in the n-th row.
        sign = - int((j % 2 - .5) * 2.)
        m = sign * ((k + 1 - (n % 2)) // 2 * 2 + (n % 2))
        return n, m

    def get_poly(self, r, p, n, m):
        """Calculates the polynomial according to (n,m) indexing."""
        if m==0:
            norm = np.sqrt(n + 1.)  # Normalization
            azim = 1.               # Azimuthal part
        else:
            norm = np.sqrt(2 * n + 2.)
            azim = np.cos(m * p) if m > 0 else np.sin(-m * p)
        m = abs(m)
        if m==n:
            rad = r ** n            # Radial part
        else:
            factor = lambda k: (-1.) ** k * fac(n - k) / (fac(k)
                                                        * fac((n + m) / 2 - k)
                                                        * fac((n - m) / 2 - k))
            rad = np.sum(factor(k) * r ** (n - 2 * k)
                                            for k in np.arange((n - m) / 2 + 1))
        return norm * rad * azim

    def make_zernikes(self, u0, v0, a):
        """Allocates a set of Zernike polynomials for a given pupil.

        If the pupil has not changed since the last call,
        no action is taken.

        Args:
            u0, v0: Pupil center coordinates in pixels.
            a: Pupil aperture radius in pixels.
        """
        if self.u0 != u0 or self.v0 != v0 or self.a != a:
            self.u0, self.v0 = u0, v0
            self.a = a
            self.r = (np.sqrt((self.u - self.u0) ** 2 + (self.v - self.v0) ** 2)
                      / self.a)
            self.p = np.arctan2(self.v - self.v0, self.u - self.u0)
            self.R = circle(self.u0, self.v0, self.a, self.N)
            for j in range(self.jmax):
                n, m = self.get_indices(j + 1)
                self.Z[j] = self.get_poly(self.r, self.p, n, m)

    def fit(self, W, u0, v0, a):
        """Calculates Zernike expansion coefficients for a wavefront.

        Args:
            W: The wavefront to be expanded into Zernike series.
            u0, v0: Pupil center coordinates in pixels.
            a: Pupil aperture radius in pixels.

        Returns:
            An array of length jmax representing the fitted
            expansion coefficients.
        """
        self.make_zernikes(u0, v0, a)
        C = (np.sum(W[None,:,:] * self.Z * self.R, axis=(-2,-1))
             / (np.pi * self.a ** 2))
        return C

    def __call__(self, C, u0, v0, a):
        """Returns a wavefront given by an array of expansion coefficients.

        Args:
            C: Array of Zernike expansion coefficients.  The length
                must not exceed jmax.
            u0, v0: Pupil center coordinates in pixels.
            a: Pupil aperture radius in pixels.

        Rreturns:
            An NxN array representing the wavefront.
        """
        self.make_zernikes(u0, v0, a)
        W = np.sum(C[:,None,None] * self.Z, axis=0)
        return W
