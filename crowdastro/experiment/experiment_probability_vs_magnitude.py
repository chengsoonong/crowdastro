"""Plots the magnitude features against predicted probability.

Matthew Alger
The Australian National University
2016
"""

import argparse
import collections
import logging

import h5py
import matplotlib.mlab
import matplotlib.pyplot as plt
import numpy
import scipy.interpolate
import sklearn.linear_model
import sklearn.preprocessing


def main(crowdastro_h5_path, training_h5_path):
    with h5py.File(training_h5_path, 'r') as training_h5:
        features = training_h5['features'].value

        labels = training_h5['labels'].value
        lr = sklearn.linear_model.LogisticRegression(class_weight='balanced')
        lr.fit(features, labels)
        with h5py.File(crowdastro_h5_path, 'r') as crowdastro_h5:
            probs = crowdastro_h5['/wise/cdfs/norris_labels'].value
        # probs = lr.predict_proba(features)[:, 1]

        features[:, 4] = -2.5 * numpy.log10(features[:, 4])
        features[:, 5] = -2.5 * numpy.log10(features[:, 5])

        # Downsample.
        indices = numpy.arange(features.shape[0])
        numpy.random.shuffle(indices)
        indices = indices[:len(indices) // 4]

        w1_w2 = features[indices, 4]
        w2_w3 = features[indices, 5]

        xy_to_z = {}
        xy_to_n = collections.defaultdict(int)
        for x, y, z in zip(w2_w3, w1_w2, probs):
            if (x, y) in xy_to_z:
                xy_to_z[x, y] = (xy_to_z[x, y] * xy_to_n[x, y] + z) / (
                    xy_to_n[x, y] + 1)
                xy_to_n[x, y] += 1
            else:
                xy_to_z[x, y] = z
                xy_to_n[x, y] = 1

        xs, ys, zs = [], [], []
        for (x, y), z in xy_to_z.items():
            xs.append(x)
            ys.append(y)
            zs.append(z)

        x_min, x_max = numpy.percentile(xs, (2, 98))
        y_min, y_max = numpy.percentile(ys, (2, 98))
        res = 100
        xi = numpy.linspace(x_min, x_max, res)
        yi = numpy.linspace(y_min, y_max, res)
        xi, yi = numpy.meshgrid(xi, yi)
        rbf = scipy.interpolate.Rbf(xs, ys, zs, function='linear')
        zi = rbf(xi, yi)
        plt.pcolormesh(xi, yi, zi)
        plt.show()

        # plt.subplot(2, 2, 1)
        # plt.scatter(features[labels == 0, 4], probs[labels == 0],
        #             color='red', marker='+')
        # plt.scatter(features[labels == 1, 4], probs[labels == 1],
        #             color='blue', marker='+')
        # plt.xlabel('w1 - w2')
        # plt.ylabel('$p(z \\mid x)$')
        # plt.ylim((0, 1))
        # plt.subplot(2, 2, 2)
        # plt.scatter(features[labels == 0, 5], probs[labels == 0],
        #             color='red', marker='+')
        # plt.scatter(features[labels == 1, 5], probs[labels == 1],
        #             color='blue', marker='+')
        # plt.xlabel('w2 - w3')
        # plt.ylabel('$p(z \\mid x)$')
        # plt.ylim((0, 1))
        # plt.subplot(2, 2, 3)
        # plt.scatter(features[labels == 0, 6], probs[labels == 0],
        #             color='red', marker='+')
        # plt.scatter(features[labels == 1, 6], probs[labels == 1],
        #             color='blue', marker='+')
        # plt.xlabel('Distance')
        # plt.ylabel('$p(z \\mid x)$')
        # plt.ylim((0, 1))
        # plt.subplot(2, 2, 4)
        # plt.scatter(features[labels == 0, 8], probs[labels == 0],
        #             color='red', marker='+')
        # plt.scatter(features[labels == 1, 8], probs[labels == 1],
        #             color='blue', marker='+')
        # plt.xlabel('CNN2')
        # plt.ylabel('$p(z \\mid x)$')
        # plt.ylim((0, 1))
        # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crowdastro', default='data/crowdastro.h5',
                        help='HDF5 crowdastro data file')
    parser.add_argument('--training', default='data/training.h5',
                        help='HDF5 training data file')
    args = parser.parse_args()

    logging.root.setLevel(logging.INFO)

    main(args.crowdastro, args.training)
