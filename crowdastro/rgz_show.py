"""Visualisations of Radio Galaxy Zoo subjects."""

import astropy.io.fits
import matplotlib.colors
import matplotlib.pyplot
import numpy

from . import data
from .config import config

def image(im, contrast=0.05):
    """Plots an RGZ image.

    im: NumPy array of an RGZ image.
    contrast: Log scale parameter, default 0.05.
    -> MatPlotLib image plot.
    """
    im = im - im.min() + contrast
    return matplotlib.pyplot.imshow(im, origin='lower', cmap='gray',
            norm=matplotlib.colors.LogNorm(vmin=im.min(), vmax=im.max()))

def clicks(cs, colour='gray'):
    """Plots a list of RGZ clicks.

    Clicks will be flipped and scaled to match the FITS images.

    cs: List of (x, y) click tuples.
    -> MatPlotLib scatter plot.
    """
    cs = (config['surveys']['atlas']['fits_height'] -
          numpy.array(cs) * config['surveys']['atlas']['click_to_fits'])
    return matplotlib.pyplot.scatter(cs[:, 0], cs[:, 1], color=colour)

def contours(subject, colour='gray'):
    """Plots the contours of a subject.

    subject: RGZ subject.
    colour: Colour to plot contours in. Default 'gray'.
    """
    for row in data.get_contours(subject)['contours']:
        for col in row:
            xs = []
            ys = []
            for pair in col['arr']:
                xs.append(pair['x'])
                ys.append(pair['y'])
            ys = config['surveys']['atlas']['fits_height'] - numpy.array(ys)
            matplotlib.pyplot.plot(xs, ys, c=colour)

def ir(subject):
    """Plots the IR image of a subject.

    subject: RGZ subject.
    -> MatPlotLib image plot.
    """
    return image(data.get_ir(subject))

def radio(subject):
    """Plots the radio image of a subject.

    subject: RGZ subject.
    -> MatPlotLib image plot.
    """
    return image(data.get_radio(subject))

def subject(s):
    """Shows the IR and contours of a subject.

    s: RGZ subject.
    """
    ir(s)
    contours(s, colour='green')
    matplotlib.pyplot.xlim(0, config['surveys']['atlas']['fits_width'])
    matplotlib.pyplot.ylim(0, config['surveys']['atlas']['fits_height'])
