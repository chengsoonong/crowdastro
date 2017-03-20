"""Builds the convolutional neural network model.

Matthew Alger
The Australian National University
2016
"""

import argparse


def main(n_filters, conv_size, pool_size, dropout,
         patch_size, out_path=None):
    # Imports must be in the function, or whenever we import this module, Keras
    # will dump to stdout.
    import keras.layers.core as core
    import keras.layers.convolutional as conv
    import keras.models as models

    im_in = core.Input(shape=(1, patch_size, patch_size))
    conv1 = conv.Convolution2D(n_filters, conv_size, conv_size,
                               border_mode='valid',
                               activation='relu')(im_in)
    pool1 = conv.MaxPooling2D(pool_size=(pool_size, pool_size))(conv1)
    conv2 = conv.Convolution2D(n_filters, conv_size, conv_size,
                               border_mode='valid',
                               activation='relu')(pool1)
    pool2 = conv.MaxPooling2D(pool_size=(pool_size, pool_size))(conv2)
    conv3 = conv.Convolution2D(n_filters, conv_size, conv_size,
                               border_mode='valid', activation='relu')(pool2)
    dropout = core.Dropout(dropout)(conv3)
    flatten = core.Flatten()(dropout)
    lr = core.Dense(1, activation='sigmoid')(flatten)

    model = models.Model(inputs=im_in, outputs=lr)
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
    parser.add_argument('--patch_size', help='size of image patches',
                        default=32, type=int)


def _main(args):
    main(args.n_filters, args.conv_size, args.pool_size, args.dropout,
         args.patch_size, out_path=args.out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    _populate_parser(parser)
    args = parser.parse_args()
    _main(args)
