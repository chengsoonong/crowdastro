"""Trains classifiers.

Matthew Alger
The Australian National University
2016
"""

import argparse
import logging

import h5py
import numpy
import sklearn.externals.joblib
import sklearn.ensemble
import sklearn.kernel_approximation
import sklearn.linear_model
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.svm
import sklearn.metrics
import sknn.mlp
import unbalanced_dataset.over_sampling.random_over_sampler as ros

from .config import config

PATCH_RADIUS = config['patch_radius']
PATCH_DIAMETER = PATCH_RADIUS * 2

def train(inputs_h5, training_h5, classifier_out_path,
          astro_transformer_out_path, image_transformer_out_path,
          classifier='lr', use_astro=True, use_cnn=True, n_jobs=-1):
    """Trains logistic regression.

    inputs_h5: Inputs HDF5 file.
    training_h5: Training HDF5 file.
    classifier_out_path: Output classifier path.
    astro_transformer_out_path: Output astro transformer path.
    image_transformer_out_path: Output image transformer path.
    classifier: Classifier to use in {'lr', 'rf'}. Default 'lr'.
    use_astro: Use astronomical features. Default True.
    use_cnn: Use CNN features. Default True.
    n_jobs: Number of cores to use. Default all.
    -> classifier
    """
    if not any([use_astro, use_cnn]):
        raise ValueError('Must have features.')

    training_indices = (training_h5['sets'][:, 0] == 1).nonzero()[0]
    outputs = training_h5['labels'].value

    astro_inputs = numpy.minimum(training_h5['astro'].value, 1500)
    image_inputs = training_h5['cnn_outputs'].value

    astro_transformer = sklearn.pipeline.Pipeline([
            ('normalise', sklearn.preprocessing.Normalizer()),
            ('scale', sklearn.preprocessing.StandardScaler()),
    ])
    image_transformer = sklearn.pipeline.Pipeline([
            ('normalise', sklearn.preprocessing.Normalizer())
    ])

    features = []
    if use_astro:
        features.append(astro_transformer.fit_transform(astro_inputs))
    if use_cnn:
        features.append(image_transformer.fit_transform(image_inputs))
    inputs = numpy.hstack(features)

    inputs = inputs[training_indices]
    outputs = outputs[training_indices]

    if classifier == 'lr':
        classifier = sklearn.linear_model.LogisticRegression(
                class_weight='balanced', n_jobs=n_jobs)
    elif classifier == 'rf':
        classifier = sklearn.ensemble.RandomForestClassifier(
                class_weight='balanced', n_jobs=n_jobs)
    elif classifier == 'nn':
        classifier = sknn.mlp.Classifier(
                layers=[sknn.mlp.Layer('Sigmoid', units=inputs.shape[1] // 2),
                        sknn.mlp.Layer('Softmax')],
                # Match sklearn parameters. They're default for sklearn, but
                # not for sknn.
                n_iter=100,
                regularize='L2')
        sampler = ros.RandomOverSampler()
        sampler.fit(inputs, outputs)
        inputs, outputs = sampler.transform(inputs, outputs)
    elif classifier == 'svm':
        classifier = sklearn.svm.SVC(class_weight='balanced', probability=True)
    elif classifier == 'klr':
        # A bit of a hack to reduce the size of the data so we can guess a gamma
        # value. Otherwise we run out of memory!
        # TODO(MatthewJA): This will break for bigger data sets.
        input_sample, _ = sklearn.cross_validation.train_test_split(
                inputs, train_size=0.2, stratify=outputs)
        dists = sklearn.metrics.pairwise_distances(input_sample, n_jobs=n_jobs)
        gamma = 1 / numpy.percentile(dists ** 2, 75)
        sampler = sklearn.kernel_approximation.RBFSampler(gamma=gamma,
                n_components=100)
        lr = sklearn.linear_model.LogisticRegression(
                class_weight='balanced', n_jobs=n_jobs)
        classifier = sklearn.pipeline.Pipeline([
                ("feature_map", sampler),
                ("lr", lr)])
    else:
        raise ValueError('Unknown classifier: {}'.format(classifier))

    classifier.fit(inputs, outputs)

    sklearn.externals.joblib.dump(classifier, classifier_out_path)
    sklearn.externals.joblib.dump(astro_transformer, astro_transformer_out_path)
    sklearn.externals.joblib.dump(image_transformer, image_transformer_out_path)

    return classifier


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', default='crowdastro.h5',
                        help='HDF5 input data file')
    parser.add_argument('--training', default='training.h5',
                        help='HDF5 training data file')
    parser.add_argument('--c_out', default='classifier.pkl',
                        help='classifier output file')
    parser.add_argument('--at_out', default='astro_transformer.pkl',
                        help='astro_transformer output file')
    parser.add_argument('--it_out', default='image_transformer.pkl',
                        help='image_transformer output file')
    parser.add_argument('--classifier',
                        choices={'lr', 'rf', 'nn', 'svm', 'klr'},
                        default='lr', help='which classifier to train')
    parser.add_argument('--no_astro', action='store_false', default=True,
                        help='ignore astro features')
    parser.add_argument('--no_cnn', action='store_false', default=True,
                        help='ignore CNN features')
    parser.add_argument('--n_jobs', default=-1, type=int,
                        help='number of cores to use')
    args = parser.parse_args()

    logging.root.setLevel(logging.DEBUG)

    with h5py.File(args.training, 'r') as training_h5:
        with h5py.File(args.inputs, 'r') as inputs_h5:
            train(inputs_h5, training_h5, args.c_out, args.at_out, args.it_out,
                  classifier=args.classifier, use_astro=args.no_astro,
                  use_cnn=args.no_cnn)
