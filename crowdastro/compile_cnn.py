"""Builds the convolutional neural network model.

Matthew Alger
The Australian National University
2016
"""

import argparse


def main(n_filters, conv_size, pool_size, dropout, hidden_layer_size,
         patch_size, out_path=None):
    # Imports must be in the function, or whenever we import this module, Keras
    # will dump to stdout.
    import keras.layers.core as core
    import keras.layers.convolutional as conv
    import keras.models as models

    model = models.Sequential()

    model.add(conv.Convolution2D(n_filters, conv_size, conv_size,
                                 border_mode='valid',
                                 input_shape=(1, patch_size, patch_size)))
    model.add(core.Activation('relu'))
    model.add(conv.MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(conv.Convolution2D(n_filters, conv_size, conv_size,
                                 border_mode='valid',))
    model.add(core.Activation('relu'))
    model.add(conv.MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(conv.Convolution2D(n_filters, conv_size, conv_size,
                                 border_mode='valid',))
    model.add(core.Activation('relu'))
    model.add(core.Dropout(dropout))
    model.add(core.Flatten())
    model.add(core.Dense(hidden_layer_size))
    model.add(core.Activation('sigmoid'))
    model.add(core.Dense(1))
    model.add(core.Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adadelta')

    model_json = model.to_json()

    if out_path is not None:
        with open(out_path, 'w') as f:
            f.write(model_json)

    return model_json


def _populate_parser(parser):
    parser.description = 'Compiles a convolutional neural network to JSON.'
    parser.add_argument('--out_path', help='path to output model JSON',
                        default='data/model.json')
    parser.add_argument('--n_filters', help='number of convolutional filters',
                        default=32, type=int)
    parser.add_argument('--conv_size', help='size of convolutional filters',
                        default=4, type=int)
    parser.add_argument('--pool_size', help='size of max pool',
                        default=2, type=int)
    parser.add_argument('--dropout', help='dropout percentage',
                        default=0.25, type=float)
    parser.add_argument('--hidden_layer_size', help='hidden layer size',
                        default=64, type=int)
    parser.add_argument('--patch_size', help='size of image patches',
                        default=32, type=int)


def _main(args):
    main(args.n_filters, args.conv_size, args.pool_size, args.dropout,
         args.hidden_layer_size, args.patch_size, out_path=args.out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    _populate_parser(parser)
    args = parser.parse_args()
    _main(args)
