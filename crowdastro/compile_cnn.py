"""Builds the convolutional neural network model.

Matthew Alger
The Australian National University
2016
"""

import argparse


def main(n_filters, conv_size, pool_size, dropout,
         patch_size, n_astro=7, out_path=None):
    # Imports must be in the function, or whenever we import this module, Keras
    # will dump to stdout.
    import keras.layers.core as core
    from keras.layers import Input, Dense
    import keras.layers.convolutional as conv
    import keras.layers.merge
    from keras.models import Model

    im_in = Input(shape=(1, patch_size, patch_size))
    astro_in = Input(shape=(n_astro,))
    # 1 x 32 x 32
    conv1 = conv.Convolution2D(filters=n_filters,
                               kernel_size=(conv_size, conv_size),
                               border_mode='valid',
                               activation='relu',
                               data_format='channels_first')(im_in)
    # 32 x 28 x 28
    pool1 = conv.MaxPooling2D(pool_size=(pool_size, pool_size),
                              data_format='channels_first')(conv1)
    # 32 x 14 x 14
    conv2 = conv.Convolution2D(filters=n_filters,
                               kernel_size=(conv_size, conv_size),
                               border_mode='valid',
                               activation='relu',
                               data_format='channels_first')(pool1)
    # 32 x 10 x 10
    pool2 = conv.MaxPooling2D(pool_size=(pool_size, pool_size),
                              data_format='channels_first')(conv2)
    # 32 x 5 x 5
    conv3 = conv.Convolution2D(filters=n_filters,
                               kernel_size=(conv_size, conv_size),
                               border_mode='valid', activation='relu',
                               data_format='channels_first')(pool2)
    # 32 x 1 x 1
    dropout = core.Dropout(dropout)(conv3)
    flatten = core.Flatten()(dropout)
    conc = keras.layers.merge.Concatenate()([astro_in, flatten])
    lr = Dense(1, activation='sigmoid')(conc)

    model = Model(inputs=[astro_in, im_in], outputs=[lr])
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
