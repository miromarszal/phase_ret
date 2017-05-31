# -*- coding: utf-8 -*-
"""phase_ret3.py

Utility functions and classes for phase retrieval, image analysis
and Zernike polynomial fitting.
An improved version of phase_ret.py transferred to Python 3.

author: Miroslaw Marszalek
"""

# %%
import numpy as np
from scipy.ndimage import extrema
from scipy.special import j1
from scipy.signal import resample
from pandas import DataFrame
import sys
fac = np.math.factorial
# %%

# Optionally import the tifffile module. It provides the Tiff stack
# functionality.  If not available, Tiff stacks will have to be
# loaded into Numpy arrays another way.
try:
    import tifffile
    TIFF_LOADED = True
except ImportError:
    TIFF_LOADED = False
    print('Failed to import tifffile. Store Tiff stacks in Numpy arrays.')


def circle(x0, y0, r, L):
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
        L: Width of the image to be returned.

    Returns:
        An LxL numpy.array with values of 1 inside the circle,
        0 outside of it, and weighted accordingly on the edge.
    """
    L0 = int((np.ceil(r)+1)*2)  # Subarray size
    dx = int(x0-L0/2)           # Number of pixels to pad to the full array
    dy = int(y0-L0/2)
    circ = np.zeros((L, L))

    # Loop over the subarray
    for j, i in np.ndindex((L0, L0)):
        d = 2. * (np.sqrt((x0-dx-i)**2 + (y0-dy-j)**2) - r)
        # Pixel outside the circle
        if d >= 1:
            circ[dy+j, dx+i] = 0.
        # Pixel inside the circle
        elif d <= -1:
            circ[dy+j, dx+i] = 1.
        # Intermediate case - edge pixels
        else:
            circ[dy+j, dx+i] = (np.arccos(d) - d*np.sqrt(1.-d**2)) / np.pi

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
    x0 = round(x0) - s // 2
    y0 = round(y0) - s // 2
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


def total_power(img, x0, y0, r1=100, r2=250, r3=300, Xi=None, Yi=None):
    """Calculates the total power of an image in a circular region.

    The algorithm sums all the pixel values from the image inside a
    circle of radius r1 centred on (x0, y0) and subtracts background
    taken as the mean value of the annulus between r2 and r3 centred
    on (x0, y0).

    Args:
        img: An image to be analyzed (numpy.array).
        x0, y0: Peak coordinates.
        r1, r2, r3: Radii for sectioning the image.
        Xi, Yi: Pixel coordnate arrays, to speed up the calculation.

    Returns a dict of results:
        P: The total power.
        var_P: Variance of the total power value.
        Nsig: Number of signal pixels used in the calculation.
        bg: The background level.
        var_bg: Variance of the background level.
        Nbg: Number of background pixels used in the calculation.
    """
    if Xi is None or Yi is None:
        Yi, Xi = np.indices(img.shape)

    # Sectioning the image into the signal and background parts
    img_sig = img[(Xi - x0) ** 2 + (Yi - y0) ** 2 <= r1 ** 2]
    img_bg = img[((Xi - x0) ** 2 + (Yi - y0) ** 2 <= r3 ** 2)
                 * ((Xi - x0) ** 2+(Yi - y0) ** 2 >= r2 ** 2)]
    Nsig = img_sig.size
    Nbg = img_bg.size

    # Estimating the background and the total power
    bg = img_bg.mean()
    var_bg = img_bg.var(ddof=1)
    P = img_sig.sum() - Nsig * bg
    var_P = P + Nsig * (1. + 1. / Nbg) * var_bg

    return {'P': P, 'var_P': var_P, 'Nsig': Nsig, 'bg': bg,
            'var_bg': var_bg, 'Nbg': Nbg}


def analyze_peaks(stack, window, res,
                  r1=100, r2=250, r3=300, print_output=True):
    """Finds peaks in a Tiff stack and does several measurements.

    Accepts a TiffFile stack and performs image analysis on it,
    returning a set of measurements for each page in the stack.
    Peak location (x0, y0) is extracted with sub-pixel accuracy by
    resampling with the Fourier method in a given window around the
    maximum and reading out the maximum position.  Background and
    total power are estimated via the total_power function (see
    below).  The amplitude from the resampled image is then divided
    by the total power to yield the normalized amplitude.

    Args:
        stack: A stack of images to be analyzed, either TiffFile
            or a 3D numpy.array.
        window: Size of the window used to resample an image.
        res: Resampling rate, a pixel is divided into res^2 pixels.
        r1, r2, r3: Radii for total_power (see below).
        print_output: If true, measurement results will be printed.

    Returns:
        parameters, variances : pandas.DataFrame of measurement
            results, each row in a frame corresponds to a page of
            the stack.

    Columns correspond to:
    amp        -- peak amplitude of the resampled image after
                  background subtraction
    x0, y0     -- sub-pixel peak location


    bg         -- background level
    tot power  -- total power of the signal
    norm amp   -- peak amplitude of the resampled image normalized
                  to the total power
    """
    parameters = DataFrame(columns=['amp', 'x0', 'y0', 'bg', 'tot power',
                                    'norm amp'], index=np.arange(len(stack)),
                           dtype='float64')
    variances = DataFrame(columns=['amp', 'bg', 'tot power', 'norm amp'],
                          index=np.arange(len(stack)), dtype='float64')
    xw = np.arange(window)  # Window pixel coordinates

    # Loop over all images in the stack
    for i, img in enumerate(stack):
        if TIFF_LOADED and isinstance(stack, tifffile.TiffFile):
            img = img.asarray()
        y0, x0 = extrema(img)[3]  # Coarse maximum location
        imgCp = crop(img, x0, y0, window)  # Windowing
        # Image resampling, getting the sub-pixel peak position
        imgRes, xRes = resample(resample(imgCp, window*res, xw, axis=0)[0],
                                window*res, xw, axis=1)
        y0Res, x0Res = np.unravel_index(imgRes.argmax(), imgRes.shape)
        # Converting peak coordinates from window to full image
        x0 = xRes[x0Res] + x0 - window/2
        y0 = xRes[y0Res] + y0 - window/2
        # Measurements and normalization
        P, var_P, Ns, bg, var_bg, Nbg = total_power(img, x0, y0,
                                                    r1=r1, r2=r2, r3=r3)
        Imax = imgRes.max() - bg
        var_Imax = Imax + (1. + 1./Nbg) * var_bg
        Inorm = Imax / P
        var_Inorm = Inorm**2 * (var_Imax/Imax**2 + var_P/P**2)
        # Saving to containers
        parameters.iloc[i] = Imax, x0, y0, bg, P, Inorm
        variances.iloc[i] = var_Imax, var_bg, var_P, var_Inorm
    if print_output:
        print('parameters:\n')
        print(parameters)
        print('\nvariances:\n')
        print(variances)
    return parameters, variances
