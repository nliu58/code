""" Functions to do spectral resampling and smoothing.
@author: Nanfeng Liu (nliu58@wisc.edu)
"""
import numpy as np

def estimate_fwhms_from_waves(waves):
    """ Estimate fwhm from wavelengths.
    Notes:
        (1) The code is adapted from the Hytools.
            https://github.com/EnSpec/HyTools-sandbox/blob/master/hytools/preprocess/resampling.py
    Arguments:
        waves: array
            Wavelengths, in nm.
    Returns:
        fwhms: array
            Full width at half maximum, in nm.
    """

    gap = 0.5*np.diff(waves)
    gap = gap[1:] + gap[:-1]
    fwhms = np.append(np.append(gap[0]*2, gap), gap[-1]*2)

    return fwhms

def gaussian(x, mu, fwhm):
    """ Return a gaussian distribution.
    Arguments:
        x: array
            Wavelengths along which to generate gaussian.
        mu: float
            Centre wavelength.
        fwhm: float
            Full width half maximum.
    Returns:
        Numpy array of gaussian along input range.
    """
    sigma = fwhm/(2* np.sqrt(2*np.log(2)))

    return np.exp(-1*((x-mu)**2/(2*sigma**2)))/(sigma*np.sqrt(2*np.pi))

def get_resampling_coeff(src_waves, dst_waves, dst_fwhms):
    """ Return a set of coeffiencients for spectrum resampling.
    Notes:
        (1) Given a set of source wavelengths, destination wavelengths and FWHMs this
            function caculates the relative contribution or each input wavelength
            to the output wavelength. It assumes that output
            response functions follow a gaussian distribution.
    Arguments:
        src_waves: array
            List of source wavelength centers.
        dst_waves: array
            List of destination wavelength centers.
        dst_fwhms : array
            List of destination full width half maxes.
    Returns:
        m x n matrix of coeffiecients, where m is the number of source wavelengths
        and n is the number of destination wavelengths.
    """

    dst_matrix = []
    for dst_wave, dst_fwhm in zip(dst_waves, dst_fwhms):
        a =  gaussian(src_waves -.5, dst_wave, dst_fwhm)
        b =  gaussian(src_waves +.5, dst_wave, dst_fwhm)
        area = (a+b)/2
        dst_matrix.append(np.divide(area,np.sum(area)))
    dst_matrix = np.array(dst_matrix)

    return dst_matrix.T