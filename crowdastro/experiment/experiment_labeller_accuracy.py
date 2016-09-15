"""Plots the accuracies of RGZ volunteers.

Matthew Alger
The Australian National University
2016
"""

import argparse

import crowdastro.crowd.util
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy
import sklearn.metrics


def main(crowdastro_h5_path):
    with h5py.File(crowdastro_h5_path, 'r') as f_h5:
        ir_survey = f_h5.attrs['ir_survey']
        labels = f_h5['/{}/cdfs/rgz_raw_labels'.format(ir_survey)].value
        mask = f_h5['/{}/cdfs/rgz_raw_labels_mask'.format(ir_survey)].value
        masked_labels = numpy.ma.MaskedArray(labels, mask=mask)
        norris_labels = f_h5['/{}/cdfs/norris_labels'.format(ir_survey)].value

        accuracies = []
        alphas = []  # True positive rate.
        betas = []  # 1 - false positive rate.

        for t in range(labels.shape[0]):
            mask = ~masked_labels[t].mask
            cm = sklearn.metrics.confusion_matrix(
                    norris_labels[mask], masked_labels[t][mask])
            ba = crowdastro.crowd.util.balanced_accuracy(
                    norris_labels, masked_labels[t])
            if ba:
                alphas.append(cm[1, 1] / (cm[1, 1] + cm[1, 0]))
                betas.append(1 - cm[0, 1] / (cm[0, 1] + cm[0, 0]))
                accuracies.append(ba)

        print('Average accuracy: ({} +- {})%'.format(
                numpy.mean(accuracies), numpy.std(accuracies)))
        print('Average alpha: ({} +- {})%'.format(
                numpy.mean(alphas), numpy.std(alphas)))
        print('Average beta: ({} +- {})%'.format(
                numpy.mean(betas), numpy.std(betas)))

        matplotlib.rcParams['font.family'] = 'serif'
        matplotlib.rcParams['font.serif'] = ['Palatino Linotype']

        plt.hist(accuracies, bins=20, color='grey')
        plt.xlabel('Balanced accuracy')
        plt.ylabel('Number of labellers')
        plt.show()

        plt.hist(alphas, bins=20, color='grey')
        plt.xlabel('True positive rate')
        plt.ylabel('Number of labellers')
        plt.show()
        plt.hist(betas, bins=20, color='grey')
        plt.xlabel('1 - false positive rate')
        plt.ylabel('Number of labellers')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crowdastro', default='data/crowdastro.h5',
                        help='crowdastro HDF5 file')

    args = parser.parse_args()

    main(args.crowdastro)
