"""Finds labels for ELAIS-S1 objects.

The Australian National University
2016
"""

import argparse

import h5py
import sklearn.externals.joblib
import sklearn.linear_model


def main(training_h5_path, classifier_path, output_h5_path):
    with h5py.File(training_h5_path, 'r') as f_h5:
        features = f_h5['features']
        classifier = sklearn.externals.joblib.load(classifier_path)
        predictions = classifier.predict(features)
        with h5py.File(output_h5_path, 'w') as g_h5:
            g_h5.create_dataset('predictions', data=predictions, dtype=bool)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training', default='data/training.h5')
    parser.add_argument('--classifier', default='data/classifier.dat')
    parser.add_argument('--output', default='data/predictions.h5')
    args = parser.parse_args()
    main(args.training, args.classifier, args.output)

