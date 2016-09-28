"""Shows filters of a convolutional neural network.

Matthew Alger
The Australian National University
2016
"""

import argparse

import h5py
import matplotlib.pyplot as plt
import numpy

from crowdastro.config import config


def main(model_path, weights_path, training_h5_path):
    import keras.models
    with open(model_path, 'r') as model_json:
        model = keras.models.model_from_json(model_json.read())
    model.compile(loss='binary_crossentropy', optimizer='adadelta')
    model.load_weights(weights_path)
    conv_layer_0 = model.get_weights()[0]
    conv_layer_1 = model.get_weights()[2]

    # n_filters = conv_layer_0.shape[0]
    # plt.figure(figsize=[8, 4])
    # for i, filt in enumerate(conv_layer_0):
    #     plt.subplot(4, 8, i + 1)
    #     plt.axis('off')
    #     filt = filt[0]
    #     plt.pcolor(filt, cmap='viridis')
    # plt.subplots_adjust(hspace=0.05, wspace=0.05, left=0.00, bottom=0.00,
    #                     top=1.00, right=1.00)
    # plt.show()

    # n_filters = conv_layer_1.shape[0] * conv_layer_1.shape[1]
    # plt.figure(figsize=[8, 8])
    # for i, filts in enumerate(conv_layer_1):
    #     for j, filt in enumerate(filts):
    #         plt.subplot(32, 32, i * 32 + j + 1)
    #         plt.axis('off')
    #         plt.pcolor(filt, cmap='viridis')
    # plt.subplots_adjust(hspace=0.05, wspace=0.05, left=0.00, bottom=0.00,
    #                     top=1.00, right=1.00)
    # plt.show()

    encoded_0 = keras.backend.function([model.layers[0].input],
                                       [model.layers[0].output])
    encoded_1 = keras.backend.function([model.layers[0].input],
                                       [model.layers[2].output])
    print('Weights shapes:', [i.shape for i in model.get_weights()])

    with h5py.File(training_h5_path, 'r') as f_h5:
        n_raw_features = config['surveys'][
            f_h5.attrs['ir_survey']]['n_features']
        for i in numpy.random.randint(f_h5['raw_features'].shape[0],
                                      size=10):
            img = f_h5['raw_features'][i, n_raw_features:].reshape(
                (1, 1, 32, 32))
            out_0 = encoded_0([img])[0].reshape((32, 29, 29))
            out_1 = encoded_1([img])[0].reshape((32, 14, 14))

            plt.figure(figsize=[2 + 4 + 4, 8])

            ax = plt.subplot2grid((8, 2 + 4 + 4), (3, 0), colspan=2, rowspan=2)
            ax.pcolor(img[0, 0], cmap='viridis')
            ax.axis('off')
            for j, convolved in enumerate(out_0):
                ax = plt.subplot2grid((8, 2 + 4 + 4), (j // 4, 2 + j % 4))
                plt.axis('off')
                plt.pcolor(convolved, cmap='viridis')
            for j, convolved in enumerate(out_1):
                ax = plt.subplot2grid((8, 2 + 4 + 4), (j // 4, 6 + j % 4))
                plt.axis('off')
                plt.pcolor(convolved, cmap='viridis')
            plt.show()


def _populate_parser(parser):
    parser.description = 'Shows filters of a convolutional neural network.'
    parser.add_argument('--model', help='path to model JSON',
                        default='data/model.json')
    parser.add_argument('--weights', help='path to weights HDF5',
                        default='data/weights.h5')
    parser.add_argument('--training', help='path to training HDF5',
                        default='data/training.h5')


def _main(args):
    main(args.model, args.weights, args.training)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    _populate_parser(parser)
    args = parser.parse_args()
    _main(args)
