"""Trains a convolutional neural network.

Matthew Alger
The Australian National University
2016
"""

import argparse
import logging
import subprocess

import h5py
import numpy

from .config import config

PATCH_RADIUS = config['patch_radius']
PATCH_DIAMETER = PATCH_RADIUS * 2


def train(training_h5, model_json, weights_path, epochs, batch_size, s3=False,
          s3_bucket=None, using_dataset=False, test_set_index=0):
    """Trains a CNN.

    training_h5: Training HDF5 file.
    model_json: JSON model file.
    weights_path: Output weights HDF5 file.
    epochs: Number of training epochs.
    batch_size: Batch size.
    s3: Whether to periodically dump to Amazon S3. Default False.
    s3_bucket: Name of the bucket to dump to. Must be specified iff s3 is True.
    using_dataset: Whether the given training file is the Zenodo crowdastro
        dataset. Default False (i.e. we are using the generated training file).
    test_set_index: Index of the test set to not use for training. Default 0.
    """
    if s3 and not s3_bucket:
        raise ValueError('Must specify s3_bucket to dump to S3.')
    if not s3 and s3_bucket:
        raise ValueError('s3_bucket was specified but s3 is False.')

    import keras.callbacks
    import keras.models
    from keras.preprocessing.image import ImageDataGenerator
    model = keras.models.model_from_json(model_json.read())
    model.compile(loss='binary_crossentropy', optimizer='adadelta')

    if not using_dataset:
        ir_survey = training_h5.attrs['ir_survey']
        n_nonimage_features = config['surveys'][ir_survey]['n_features']
        features_name = 'raw_features'
    else:
        ir_survey = 'wise'
        n_nonimage_features = 5
        features_name = 'features'

    test_set = set(training_h5['test_sets'][test_set_index, :])
    n_examples = training_h5[features_name].shape[0]
    train_set = [i for i in range(n_examples) if i not in test_set]
    del test_set  # Just to make sure we don't use it.

    training_inputs = training_h5[features_name].value[train_set, :]
    training_outputs = training_h5['labels'].value[train_set]
    assert training_inputs.shape[0] == training_outputs.shape[0]

    # Downsample for class balance.
    zero_indices = (training_outputs == 0).nonzero()[0]
    one_indices = (training_outputs == 1).nonzero()[0]
    subset_zero_indices = numpy.random.choice(
        zero_indices, size=(len(one_indices,)), replace=False)
    all_indices = numpy.hstack([subset_zero_indices, one_indices])
    all_indices.sort()

    training_inputs = training_inputs[all_indices]
    training_outputs = training_outputs[all_indices]
    assert (training_outputs == 1).sum() == (training_outputs == 0).sum()

    class DumpToS3(keras.callbacks.Callback):
        def __init__(self, weights_path, bucket, period=50):
            super().__init__()
            self.weights_path = weights_path
            self.bucket = bucket
            self.period = period

        def on_train_begin(self, logs={}):
            self.epochs = 0

        def on_epoch_end(self, epoch, logs={}):
            self.epochs += 1
            if self.epochs % self.period == 0:
                # Every 50 epochs...
                logging.debug('Dumping to S3...')
                res = subprocess.check_call(
                    ['aws', 's3', 'cp', self.weights_path,
                     's3://' + self.bucket + '/', '--region', 'us-east-1'])
                logging.info('Dumped to S3: {}'.format(res))

    try:
        model.load_weights(weights_path)
        logging.info('Loaded weights.')
    except OSError:
        logging.warning('Couldn\'t load weights file. Creating new file...')
        pass

    callbacks = [
        keras.callbacks.ModelCheckpoint(
                weights_path,
                monitor='val_loss',
                verbose=1,
                save_best_only=False,
                save_weights_only=True,
                mode='auto'),
    ]

    if s3:
        callbacks.append(DumpToS3(weights_path, s3_bucket))

    def create_generator(X, Y):
        """Yields generated images and auxiliary inputs.

        https://github.com/fchollet/keras/issues/3386
        """

        X_im = X[:, n_nonimage_features:].reshape(
            (-1, 1, PATCH_DIAMETER, PATCH_DIAMETER))
        X_au = X[:, :n_nonimage_features]

        while True:
            # Shuffle indices.
            idx = numpy.random.permutation(X.shape[0])
            # Standard image generator.
            datagen = ImageDataGenerator(
                    horizontal_flip=True,
                    vertical_flip=True)
            datagen.fit(X_im)
            # Shuffle the data before batching using known indices.
            batches = datagen.flow(X_im[idx], Y[idx], batch_size=batch_size,
                                   shuffle=False)
            idx0 = 0
            for batch in batches:
                idx1 = idx0 + batch[0].shape[0]

                # Yield ((image, aux), label) tuples.
                yield [batch[0], X_au[idx[idx0:idx1]]], batch[1]

                idx0 = idx1
                if idx1 >= X.shape[0]:
                    break

    model.fit_generator(create_generator(training_inputs, training_outputs),
                        steps_per_epoch=training_inputs.shape[0] // batch_size,
                        epochs=epochs,
                        callbacks=callbacks,
                        workers=8)

    model.save_weights(weights_path, overwrite=True)


def check_raw_data(training_h5):
    """Sanity check the input data

    training_h5: Training HDF5 file.
    """
    def HDF5_type(name, node):
        if isinstance(node, h5py.Dataset):
            logging.info('Dataset: {}'.format(node.name))
            logging.info('\thas shape {}'.format(node.shape))
        else:
            logging.info('\t{} of type {}'.format((node.name, type(node))))
    logging.info('Peeking into HDF5 file')
    training_h5.visititems(HDF5_type)
    logging.info('End file peeking')


def _populate_parser(parser):
    parser.description = 'Trains a convolutional neural network.'
    parser.add_argument('--h5', default='data/training.h5',
                        help='HDF5 training data file')
    parser.add_argument('--model', default='data/model.json',
                        help='JSON model file')
    parser.add_argument('--output', default='data/weights.h5',
                        help='HDF5 file for output weights')
    parser.add_argument('--epochs', default=10,
                        help='number of epochs to train for')
    parser.add_argument('--batch_size', default=100, help='batch size')
    parser.add_argument('--s3', help='dump to Amazon S3', action='store_true')
    parser.add_argument('--s3_bucket', help='name of S3 bucket', default='')
    parser.add_argument(
        '--test_set_index',
        help='Index of test set (default 0)',
        default=0,
        type=int)
    parser.add_argument('--dataset', help='training file is the Zenodo dataset',
                        action='store_true')


def _main(args):
    with h5py.File(args.h5, 'r') as training_h5:
        check_raw_data(training_h5)
        with open(args.model, 'r') as model_json:
            train(training_h5, model_json, args.output,
                  int(args.epochs), int(args.batch_size), s3=args.s3,
                  s3_bucket=args.s3_bucket,
                  test_set_index=args.test_set_index,
                  using_dataset=args.dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    _populate_parser(parser)
    args = parser.parse_args()
    _main(args)
