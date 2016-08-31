"""Plotting functions.

Matthew Alger
The Australian National University
2016
"""

import astropy.io.fits
import astropy.wcs
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy
import sklearn.metrics

from .config import config

ARCMIN = 1 / 60  # deg
FITS_CONVENTION = 1
IMAGE_DIAMETER = config['surveys']['atlas']['fits_width']

wcs = None

def _init_wcs():
    """Initialise the ATLAS image WCS. Sets global variable wcs."""
    with astropy.io.fits.open(config['data_sources']['atlas_image'],
                              ignore_blank=True) as atlas_image:
        global wcs
        wcs = astropy.wcs.WCS(atlas_image[0].header).dropaxis(3).dropaxis(2)


def ra_dec_to_pixels(subject_coords, coords):
    if wcs is None:
        _init_wcs()

    offset, = wcs.all_world2pix([subject_coords], FITS_CONVENTION)
    # The coords are of the middle of the subject.
    coords = wcs.all_world2pix(coords, FITS_CONVENTION)
    coords -= offset
    
    coords[:, 0] /= config['surveys']['atlas']['mosaic_scale_x'] * 424 / 200
    coords[:, 1] /= config['surveys']['atlas']['mosaic_scale_y'] * 424 / 200
    
    coords += [40, 40]
    
    return coords


def plot_points_on_background(points, background, noise=False, base_size=200):
    plt.imshow(background, cmap='gray')

    colours = cm.rainbow(numpy.linspace(0, 1, len(points)))
    for colour, (x, y) in zip(colours, points):
        if noise:
            x += numpy.random.normal(scale=0.5)
            y += numpy.random.normal(scale=0.5)
        plt.scatter(x, y, marker='x', c=colour, s=base_size)
    plt.axis('off')
    plt.xlim((0, background.shape[0]))
    plt.ylim((0, background.shape[1]))


def plot_classifications(atlas_vector, ir_matrix, labels, base_size=200,
                         noise=False):
    image = atlas_vector[2 : 2 + IMAGE_DIAMETER ** 2].reshape(
            (IMAGE_DIAMETER, IMAGE_DIAMETER))[60:140, 60:140]
    radio_coords = atlas_vector[:2]

    nearby = atlas_vector[2 + IMAGE_DIAMETER ** 2:] < ARCMIN
    nearby_labels = labels[nearby]

    ir_coords_ = ir_matrix[nearby, :2][nearby_labels.astype(bool)]
    ir_coords_ = ra_dec_to_pixels(radio_coords, ir_coords_)

    ir_coords = []
    for label, coords in zip(nearby_labels.nonzero()[0], ir_coords_):
        label = nearby_labels[label]
        ir_coords.extend([coords] * label)  # For label multiplicity.
    
    plot_points_on_background(ir_coords, image, base_size=base_size,
                              noise=noise)


def plot_classifications_row(atlas_vector, ir_matrix, classifier_labels,
                             rgz_labels, norris_labels, base_size=200,
                             noise=False):
    plt.subplot(1, 3, 1)
    plt.title('Classifier')
    plot_classifications(atlas_vector, ir_matrix, classifier_labels,
                         base_size=base_size, noise=noise)
    
    plt.subplot(1, 3, 2)
    plt.title('RGZ')
    plot_classifications(atlas_vector, ir_matrix, rgz_labels,
                         base_size=base_size, noise=noise)
    
    plt.subplot(1, 3, 3)
    plt.title('Norris')
    plot_classifications(atlas_vector, ir_matrix, norris_labels,
                         base_size=base_size, noise=noise)


def vertical_scatter(xs, ys, style='bx', rotation='horizontal'):
    """Plots a vertical scatter plot.

    xs: List of x labels.
    ys: List of lists of points to scatter vertically.
    style: Plots point style. Default 'bx'.
    rotation: x label rotation. Default 'horizontal'.
    """
    for x in range(len(xs)):
        plt.plot([x] * len(ys[x]), ys[x], style)
    plt.xticks(range(len(xs)), xs, rotation=rotation)


def vertical_scatter_ba(results, targets, ylim=(0.7, 1.0), **kwargs):
    """Plot a vertical scatter plot of balanced accuracies.

    results: Results object.
    targets: Target labels.
    ylim: (lower, upper) y axis.
    kwargs: Keyword arguments passed to vertical_scatter.
    """
    xs = sorted(results.methods, key=results.methods.get)
    ys = []
    for method in xs:
        y = []
        for split in range(results.n_splits):
            mask = results.get_mask(method, split)
            split_results = results[method, split][mask].round()
            split_targets = targets[mask]
            if len(split_results) == 0:
                continue
            # Calculate balanced accuracy.
            cm = sklearn.metrics.confusion_matrix(split_targets, split_results)
            tp = cm[1, 1]
            n, p = cm.sum(axis=1)
            tn = cm[0, 0]
            ba = (tp / p + tn / n) / 2
            y.append(ba)
        ys.append(y)

    vertical_scatter(xs, ys, **kwargs)
    plt.ylim(ylim)
    plt.xlim((-0.5, len(xs) - 0.5))  # Adds a little buffer.
    plt.ylabel('Balanced accuracy (%)')
