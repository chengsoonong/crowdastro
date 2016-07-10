"""Classifier for radio subjects.

Matthew Alger
The Australian National University
2016
"""

import logging

import numpy
import sklearn.cross_validation
from sklearn.ensemble import RandomForestClassifier
import sklearn.linear_model
import sklearn.pipeline
import sklearn.preprocessing

from .config import config

ARCMIN = 1 / 60  # deg
IMAGE_SIZE = (config['surveys']['atlas']['fits_height']
             * config['surveys']['atlas']['fits_width'])


class RGZClassifier(object):
    
    def __init__(self, ir_features, n_astro,
                 Classifier=sklearn.linear_model.LogisticRegression):
        self.ir_features = ir_features
        self.n_astro = n_astro
        self._classifier = Classifier(class_weight='balanced')
        self._astro_transformer = sklearn.pipeline.Pipeline([
            ('normalise', sklearn.preprocessing.Normalizer()),
            ('scale', sklearn.preprocessing.StandardScaler()),
        ])
        self._image_transformer = sklearn.pipeline.Pipeline([
            ('normalise', sklearn.preprocessing.Normalizer()),
        ])
    
    def _fit_transformers(self, ir_indices):
        self._astro_transformer.fit(self.ir_features[ir_indices, :self.n_astro])
        self._image_transformer.fit(self.ir_features[ir_indices, self.n_astro:])
    
    def _transform(self, features):
        return numpy.hstack([
            self._astro_transformer.transform(features[:, :self.n_astro]),
            self._image_transformer.transform(features[:, self.n_astro:]),
        ])
    
    def train(self, ir_indices, ir_labels):
        ir_indices = list(ir_indices)
        self._fit_transformers(ir_indices)
        ir_features = self._transform(self.ir_features[ir_indices])
        self._classifier.fit(ir_features, ir_labels)
    
    def _predict(self, atlas_vector):
        # Split the ATLAS vector into its components.
        position = atlas_vector[:2]
        image = atlas_vector[2:2 + IMAGE_SIZE]
        distances = atlas_vector[2 + IMAGE_SIZE:]
        # Get nearby IR objects and their features.
        nearby_indices = list((distances < ARCMIN).nonzero()[0])
        ir_features = self._transform(self.ir_features[nearby_indices])
        # Find how likely each object is to be the host galaxy.
        probabilities = self._classifier.predict_proba(ir_features)[:, 1]
        return nearby_indices, probabilities
    
    def predict(self, atlas_vector):
        nearby_indices, probabilities = self._predict(atlas_vector)
        # Return the index of the most likely host galaxy.
        return nearby_indices[probabilities.argmax()]
    
    def predict_probabilities(self, atlas_vector):
        nearby_indices, probabilities = self._predict(atlas_vector)
        out_probabilities = numpy.zeros(self.ir_features.shape[0])
        out_probabilities[nearby_indices] = probabilities
        return out_probabilities


def generate_subset(p, indices):
    indices = indices.copy()
    numpy.random.shuffle(indices)
    return indices[:int(p * len(indices))]


class RGZCommittee(object):

    def __init__(self, n_committee, ir_features, n_astro, lr_frac):
        self.classifiers = [RGZClassifier(ir_features, n_astro)
                            for _ in range(round(lr_frac * n_committee))]
        self.classifiers += [RGZClassifier(ir_features, n_astro,
                                           Classifier=RandomForestClassifier)
                            for _ in range(round((1 - lr_frac) * n_committee))]
        assert len(self.classifiers) == n_committee

    def train(self, ir_indices, ir_labels):
        for classifier in self.classifiers:
            classifier.train(ir_indices, ir_labels)

    def k_fold_train_and_label(self, n_folds, subset_size, atlas_vectors,
                               ir_labels):
        folds = sklearn.cross_validation.KFold(atlas_vectors.shape[0],
                                               n_folds=n_folds,
                                               shuffle=True,
                                               random_state=0)
        classifications = numpy.zeros((atlas_vectors.shape[0],
                                       len(self.classifiers)))
        for fold_index, (train_indices, test_indices) in enumerate(folds):
            logging.debug('Testing fold {}/{}.'.format(fold_index + 1, n_folds))
            # Precompute which IR objects should be reserved for testing.
            # See later comment.
            test_ir_indices = set()
            for atlas_index in test_indices:
                distances = atlas_vectors[atlas_index, 2 + IMAGE_SIZE:]
                nearby_indices = (distances < ARCMIN).nonzero()[0]
                for ir_index in nearby_indices:
                    test_ir_indices.add(ir_index)

            for index, classifier in enumerate(self.classifiers):
                train_subset = generate_subset(subset_size, train_indices)
                # This subset is of ATLAS indicies, so we need to convert this
                # into IR indices. An IR object is in the training set if it is
                # nearby a training ATLAS object and *not* nearby a testing
                # ATLAS object. The easiest way to do this is to find all nearby
                # indices, and all testing indices, and then find the set
                # difference.
                train_ir_indices = set()
                for atlas_index in train_subset:
                    distances = atlas_vectors[atlas_index, 2 + IMAGE_SIZE:]
                    nearby_indices = (distances < ARCMIN).nonzero()[0]
                    for ir_index in nearby_indices:
                        train_ir_indices.add(ir_index)
                
                train_ir_indices -= test_ir_indices
                train_ir_indices = numpy.array(sorted(train_ir_indices))

                classifier.train(train_ir_indices, ir_labels[train_ir_indices])
                
                # Classify all the test subjects.
                for atlas_index in test_indices:
                    classification = classifier.predict(
                            atlas_vectors[atlas_index])
                    classifications[atlas_index, index] = classification

        return classifications
