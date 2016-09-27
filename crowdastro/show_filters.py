"""Shows filters of a convolutional neural network.

Matthew Alger
The Australian National University
2016
"""

import argparse

import matplotlib.pyplot as plt
import numpy


def main(model_path, weights_path):
    import keras.models
    with open(model_path, 'r') as model_json:
        model = keras.models.model_from_json(model_json.read())
    model.compile(loss='binary_crossentropy', optimizer='adadelta')
    model.load_weights(weights_path)
    conv_layer_0 = model.get_weights()[0]
    conv_layer_1 = model.get_weights()[2]

    n_filters = conv_layer_0.shape[0]
    plt.figure(figsize=[8, 4])
    for i, filt in enumerate(conv_layer_0):
        plt.subplot(n_filters // 8, 8, i + 1)
        plt.axis('off')
        filt = filt[0]
        plt.pcolor(filt, cmap='viridis')
    plt.subplots_adjust(hspace=0.05, wspace=0.05, left=0.00, bottom=0.00,
                        top=1.00, right=1.00)
    plt.savefig('filters_0.pdf')

    n_filters = conv_layer_1.shape[0] * conv_layer_1.shape[1]
    plt.figure(figsize=[8, 8])
    for i, filts in enumerate(conv_layer_1):
        for j, filt in enumerate(filts):
            plt.subplot(n_filters // 32, 32, i * 32 + j + 1)
            plt.axis('off')
            plt.pcolor(filt, cmap='viridis')
    plt.subplots_adjust(hspace=0.05, wspace=0.05, left=0.00, bottom=0.00,
                        top=1.00, right=1.00)
    plt.savefig('filters_1.pdf')



def _populate_parser(parser):
    parser.description = 'Shows filters of a convolutional neural network.'
    parser.add_argument('--model', help='path to model JSON',
                        default='data/model.json')
    parser.add_argument('--weights', help='path to weights HDF5',
                        default='data/weights.h5')


def _main(args):
    main(args.model, args.weights)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    _populate_parser(parser)
    args = parser.parse_args()
    _main(args)
