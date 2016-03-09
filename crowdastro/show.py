"""Visualisations of Radio Galaxy Zoo data."""

import astropy.io.fits
import matplotlib.colors
import matplotlib.pyplot
import numpy

import config
import data

def image(im, contrast=0.05):
    """Plots an RGZ image.

    im: NumPy array of an RGZ image.
    contrast: Log scale parameter, default 0.05.
    -> MatPlotLib image plot.
    """
    im = im - im.min() + contrast
    return matplotlib.pyplot.imshow(im, origin='lower', cmap='gray',
            norm=matplotlib.colors.LogNorm(vmin=im.min(), vmax=im.max()))

def clicks(cs):
    """Plots a list of RGZ clicks.

    Clicks will be flipped and scaled to match the FITS images.

    cs: List of (x, y) click tuples.
    -> MatPlotLib scatter plot.
    """
    cs = (config.get('fits_image_height') -
          numpy.array(cs) * config.get('click_to_fits'))
    return matplotlib.pyplot.scatter(cs[:, 0], cs[:, 1])

def ir(subject):
    """Plots the IR image of a subject.

    subject: RGZ subject.
    -> MatPlotLib image plot.
    """
    return image(data.get_ir(s))

def radio(subject):
    """Plots the radio image of a subject.

    subject: RGZ subject.
    -> MatPlotLib image plot.
    """
    return image(data.get_radio(s))
