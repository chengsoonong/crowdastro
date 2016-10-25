"""Plotting functions.

Matthew Alger
The Australian National University
2016
"""

import logging

import astropy.io.fits
import astropy.wcs
import matplotlib
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


def vertical_scatter(xs, ys, style='bx', rotation='horizontal',
                     line=False, x_tick_offset=0, numeric_x=False):
    """Plots a vertical scatter plot.

    xs: List of x labels.
    ys: List of lists of points to scatter vertically.
    style: Plots point style. Default 'bx'.
    rotation: x label rotation. Default 'horizontal'.
    line: Draw lines between corresponding points. Default False.
    x_tick_offset: How far to offset the x tick labels. Default 0.
    numeric_x: Whether the x labels should be treated as numeric. Default False.
    """
    if not numeric_x:
        for x in range(len(xs)):
            plt.plot([x] * len(ys[x]), ys[x], style)
        if line:
            assert all(len(y) == len(ys[0]) for y in ys)
            ys_t = list(zip(*ys))
            for y in range(len(ys[0])):
                plt.plot(range(len(xs)), ys_t[y])
        plt.xticks([i + x_tick_offset for i in range(len(xs))], xs,
                   rotation=rotation)
        plt.xlim((-0.5, len(xs) - 0.5))  # Adds a little buffer.
    else:
        for xi, x in enumerate(xs):
            plt.plot([float(x)] * len(ys[xi]), ys[xi], style)
        if line:
            assert all(len(y) == len(ys[0]) for y in ys)
            ys_t = list(zip(*ys))
            for y in range(len(ys[0])):
                plt.plot(xs, ys_t[y])


def violinplot(xs, ys, rotation='horizontal', points=100, x_tick_offset=0,
               facecolour='lightgreen', edgecolour='green'):
    """Plots a vertical scatter plot.

    xs: List of x labels.
    ys: List of lists of points to scatter vertically.
    rotation: x label rotation. Default 'horizontal'.
    points: Number of points to use in the density estimate.
    x_tick_offset: How far to offset the x tick labels. Default 0.
    facecolour: Colour of the violin plots. Default light green.
    edgecolour: Colour of the violin lines. Default green.
    """
    vp = plt.violinplot(ys, showmeans=True, showextrema=False, points=points)
    # plt.violinplot has no arguments that let us set colours, so we have to do
    # it ourselves. http://stackoverflow.com/a/26291582/1105803
    for pc in vp['bodies']:
        pc.set_facecolor(facecolour)
        pc.set_edgecolor(edgecolour)
    vp['cmeans'].set_color(edgecolour)
    plt.xticks([1 + i + x_tick_offset for i in range(len(xs))],
               xs, rotation=rotation)
    plt.xlim((0.5, len(xs) + 0.5))  # Adds a little buffer.


def vertical_scatter_ba(results, targets, ylim=(70, 100), violin=False,
                        minorticks=False, percentage=True, **kwargs):
    """Plot a vertical scatter plot of balanced accuracies.

    results: Results object.
    targets: Target labels.
    ylim: (lower, upper) y axis.
    violin: Plot a violin plot instead. Default False.
    minorticks: Use minor ticks. Default False.
    percentage: Plot percentage rather than raw balanced accuracy. Default True.
    kwargs: Keyword arguments passed to vertical_scatter.
    """
    xs = sorted(results.method_idx, key=results.method_idx.get)
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
            if percentage:
                ba *= 100
            y.append(ba)
        logging.info('Average balanced accuracy ({}): {:.02%}'.format(
                method, numpy.mean(y)))
        logging.info('Standard deviation ({}): {:.02%}'.format(
                method, numpy.std(y)))
        ys.append(y)

    if violin:
        violinplot(xs, ys, **kwargs)
    else:
        vertical_scatter(xs, ys, **kwargs)
    plt.ylim(ylim)
    plt.grid(b=True, which='both', axis='y', color='grey', linestyle='-',
             alpha=0.5)
    if minorticks:
        plt.minorticks_on()
    plt.tick_params(axis='x', which='minor', length=0)
    plt.ylabel('Balanced accuracy' + (' (%)' if percentage else ''))


def fillbetween(xs, ys, facecolour='lightgreen', edgecolour='green',
                marker='x', facealpha=1.0, facekwargs=None, **kwargs):
    """Plots a line plot with error represented by filled-between lines.

    xs: List of x values.
    ys: List of lists of y values.
    facecolour: Colour of the filled-between lines. Default light green.
    edgecolour: Colour of the central line. Default green.
    marker: Point marker. Default 'x'.
    facealpha: Alpha value of filled section. Default 1.0.
    facekwargs: Keyword arguments to pass to fill_between. Default {}.
    """
    facekwargs = facekwargs or {}
    means = numpy.mean(ys, axis=1)
    stds = numpy.std(ys, axis=1)
    plt.plot(xs, means, color=edgecolour, marker=marker, **kwargs)
    plt.fill_between(xs, means - stds, means + stds, color=facecolour,
                     alpha=facealpha, linewidth=0, **facekwargs)
    plt.fill_between(xs, means - stds, means + stds, facecolor='None',
                     edgecolor=edgecolour, alpha=0.5, **facekwargs)
