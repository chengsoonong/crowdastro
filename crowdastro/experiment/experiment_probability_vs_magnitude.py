"""Plots the magnitude features against predicted probability.

Matthew Alger
The Australian National University
2016
"""

import argparse
import collections
import logging

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy
import seaborn
import sklearn.linear_model


def main(crowdastro_h5_path, training_h5_path):
    with h5py.File(training_h5_path, 'r') as training_h5:
        features = training_h5['features'].value

        features = sklearn.preprocessing.scale(features)

        labels = training_h5['labels'].value
        lr = sklearn.linear_model.LogisticRegression(class_weight='balanced')
        lr.fit(features, labels)
        probs = lr.predict_proba(features)[:, 1]

        plt.subplot(2, 2, 1)
        plt.scatter(features[labels == 0, 4], probs[labels == 0],
                    color='red', marker='+')
        plt.scatter(features[labels == 1, 4], probs[labels == 1],
                    color='blue', marker='+')
        plt.xlabel('w1 - w2')
        plt.ylabel('$p(z \\mid x)$')
        plt.ylim((0, 1))
        plt.subplot(2, 2, 2)
        plt.scatter(features[labels == 0, 5], probs[labels == 0],
                    color='red', marker='+')
        plt.scatter(features[labels == 1, 5], probs[labels == 1],
                    color='blue', marker='+')
        plt.xlabel('w2 - w3')
        plt.ylabel('$p(z \\mid x)$')
        plt.ylim((0, 1))
        plt.subplot(2, 2, 3)
        plt.scatter(features[labels == 0, 6], probs[labels == 0],
                    color='red', marker='+')
        plt.scatter(features[labels == 1, 6], probs[labels == 1],
                    color='blue', marker='+')
        plt.xlabel('Distance')
        plt.ylabel('$p(z \\mid x)$')
        plt.ylim((0, 1))
        plt.subplot(2, 2, 4)
        plt.scatter(features[labels == 0, 8], probs[labels == 0],
                    color='red', marker='+')
        plt.scatter(features[labels == 1, 8], probs[labels == 1],
                    color='blue', marker='+')
        plt.xlabel('CNN2')
        plt.ylabel('$p(z \\mid x)$')
        plt.ylim((0, 1))
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crowdastro', default='data/crowdastro.h5',
                        help='HDF5 crowdastro data file')
    parser.add_argument('--training', default='data/training.h5',
                        help='HDF5 training data file')
    args = parser.parse_args()

    logging.root.setLevel(logging.INFO)

    main(args.crowdastro, args.training)
