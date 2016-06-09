"""Tests classifiers on subjects.

Matthew Alger
The Australian National University
2016
"""

import argparse
import csv
import logging

import astropy.io
import astropy.wcs
import h5py
import matplotlib
import matplotlib.colors as co
import matplotlib.pyplot as plot
import numpy
import sklearn.externals.joblib
import sklearn.neighbors

from .config import config
from .data import get_ir, get_subject
from .show import contours

font = {'family' : 'Palatino Linotype',
        'size'   : 22}

matplotlib.rc('font', **font)

PATCH_RADIUS = config['patch_radius']
PATCH_DIAMETER = PATCH_RADIUS * 2
ARCMIN = 1 / 60

def test(inputs_h5, inputs_csv, training_h5, classifier_path,
         astro_transformer_path, image_transformer_path, use_astro=True,
         use_cnn=True):
    classifier = sklearn.externals.joblib.load(classifier_path)
    astro_transformer = sklearn.externals.joblib.load(astro_transformer_path)
    image_transformer = sklearn.externals.joblib.load(image_transformer_path)

    testing_indices = inputs_h5['/atlas/cdfs/testing_indices'].value
    swire_positions = inputs_h5['/swire/cdfs/catalogue'][:, :2]
    atlas_positions = inputs_h5['/atlas/cdfs/positions'].value
    all_astro_inputs = training_h5['astro'].value
    all_cnn_inputs = training_h5['cnn_outputs'].value
    all_labels = training_h5['labels'].value

    atlas_counts = {}  # ATLAS ID to number of objects in that subject.
    for consensus in inputs_h5['/atlas/cdfs/consensus_objects']:
        atlas_id = int(consensus[0])
        atlas_counts[atlas_id] = atlas_counts.get(atlas_id, 0) + 1

    simple_indices = []
    for atlas_id, count in atlas_counts.items():
        if count == 1 and atlas_id in testing_indices:
            simple_indices.append(atlas_id)
    print(simple_indices)

    atlas_positions = inputs_h5['/atlas/cdfs/positions']
    csvdicts = list(csv.DictReader(inputs_csv))
    for index, position in enumerate(atlas_positions[:100]):
        if index not in testing_indices or index not in simple_indices:
            continue

        swire_positions = training_h5['positions']
        swire_tree = sklearn.neighbors.KDTree(swire_positions)
        neighbours, distances = swire_tree.query_radius([position], ARCMIN,
                                                        return_distance=True)
        neighbours = neighbours[0]
        distances = distances[0]
        nearest_positions = swire_positions.value[neighbours]
        astro_inputs = all_astro_inputs[neighbours]
        astro_inputs[:, -1] = distances
        cnn_inputs = all_cnn_inputs[neighbours]
        labels = all_labels[neighbours]

        astro_inputs = astro_transformer.transform(astro_inputs)
        cnn_inputs = image_transformer.transform(cnn_inputs)

        probs = classifier.predict_proba(numpy.hstack([astro_inputs, cnn_inputs]))
    
        for row in csvdicts:
            if int(row['index']) == index and row['survey'] == 'atlas':
                zid = row['zooniverse_id']
                header = row['header']
                break
        subject = get_subject(zid)
        ir = get_ir(subject)
        wcs = astropy.wcs.WCS(astropy.io.fits.Header.fromstring(header))
        points = wcs.all_world2pix(nearest_positions, 1)
        print(zid)
        plot.figure(figsize=(20, 10))

        plot.subplot(1, 2, 1)
        plot.imshow(ir, cmap='gray', norm=co.LogNorm(vmin=ir.min(), vmax=ir.max()))
        contours(subject)
        plot.xlim((0, 200))
        plot.ylim((0, 200))
        plot.axis('off')
        plot.scatter(points[:, 0] - 151, points[:, 1] - 151, zorder=100, c=probs[:, 1], cmap='cool', s=100)

        plot.subplot(1, 2, 2)
        plot.scatter(range(len(probs)), sorted(probs[:, 1]), c=sorted(probs[:, 1]), cmap='cool', linewidth=0, s=100)
        plot.xlim((0, len(probs)))
        plot.ylim((0, 1))
        plot.xlabel('SWIRE object index')
        plot.ylabel('Classifier probability')
        plot.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', default='crowdastro.h5',
                        help='HDF5 inputs data file')
    parser.add_argument('--csv', default='crowdastro.csv',
                        help='CSV inputs data file')
    parser.add_argument('--training', default='training.h5',
                        help='HDF5 training data file')
    parser.add_argument('--classifier', default='classifier.pkl',
                        help='classifier file')
    parser.add_argument('--astro_transformer', default='astro_transformer.pkl',
                        help='astro transformer file')
    parser.add_argument('--image_transformer', default='image_transformer.pkl',
                        help='image transformer file')
    parser.add_argument('--no_astro', action='store_false', default=True,
                        help='ignore astro features')
    parser.add_argument('--no_cnn', action='store_false', default=True,
                        help='ignore CNN features')
    args = parser.parse_args()

    logging.root.setLevel(logging.DEBUG)

    with h5py.File(args.training, 'r') as training_h5:
        with h5py.File(args.inputs, 'r') as inputs_h5:
            with open(args.csv, 'r') as inputs_csv:
                test(inputs_h5, inputs_csv, training_h5, args.classifier,
                     args.astro_transformer, args.image_transformer,
                     use_astro=args.no_astro, use_cnn=args.no_cnn)
