"""Plots the accuracies of RGZ volunteers.

Matthew Alger
The Australian National University
2016
"""

import argparse

import crowdastro.crowd.util
import h5py
import matplotlib.pyplot as plt
import numpy


def main(crowdastro_h5_path):
    with h5py.File(crowdastro_h5_path, 'r') as f_h5:
        ir_survey = f_h5.attrs['ir_survey']
        labels = f_h5['/{}/cdfs/rgz_raw_labels'.format(ir_survey)].value
        mask = f_h5['/{}/cdfs/rgz_raw_labels_mask'.format(ir_survey)].value
        masked_labels = numpy.ma.MaskedArray(labels, mask=mask)
        norris_labels = f_h5['/{}/cdfs/norris_labels'.format(ir_survey)].value

        accuracies = []
        for t in range(labels.shape[0]):
            ba = crowdastro.crowd.util.balanced_accuracy(
                    norris_labels, masked_labels[t])
            if ba:
                accuracies.append(ba)

        print('Average accuracy: ({} +- {})%'.format(
                numpy.mean(accuracies), numpy.std(accuracies)))

        plt.hist(accuracies, bins=20)
        plt.xlabel('Balanced accuracy')
        plt.ylabel('Number of labellers')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crowdastro', default='data/crowdastro.h5',
                        help='crowdastro HDF5 file')

    args = parser.parse_args()

    main(args.crowdastro)
