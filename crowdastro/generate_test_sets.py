"""Generates sets of testing indices for the galaxy classification task.

Matthew Alger
The Australian National University
2016
"""

import argparse

import h5py
import numpy


def main(f_h5, n, p):
    ir_survey = f_h5.attrs['ir_survey']
    fan_labels = f_h5['/{}/cdfs/fan_labels'.format(ir_survey)].value
    norris_labels = f_h5['/{}/cdfs/norris_labels'.format(ir_survey)].value
    # Compute the intersection of Fan and Norris.
    intersection = (fan_labels == norris_labels).nonzero()[0]
    # Take n p% samples without replacement.
    samples = []
    for _ in range(n):
        numpy.random.shuffle(intersection)
        sample = intersection[:int(p * len(intersection))].copy()
        sample.sort()
        samples.append(sample)
    samples = numpy.array(samples)
    f_h5.create_dataset('/{}/cdfs/test_sets'.format(ir_survey),
                        data=samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5', default='data/crowdastro.h5',
                        help='Crowdastro HDF5 file')
    parser.add_argument('--n', default=5, type=int,
                        help='Number of test sets')
    parser.add_argument('--p', default=0.5, type=float,
                        help='Percentage size of test sets')

    args = parser.parse_args()

    with h5py.File(args.h5, 'r+') as f_h5:
        main(f_h5, args.n, args.p)
