"""Generates a raw features -> labels dataset.

This should be run after generate_training_data.py, but before
generate_cnn_outputs.py.

Outputs to HDF5 with structure:
/
    features
    labels

Matthew Alger
The Australian National University
2016
"""

import argparse

import h5py


def generate(f_h5, out_f_h5):
    out_f_h5.create_dataset('features', data=f_h5['features'])
    out_f_h5.create_dataset('labels', data=f_h5['labels'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', default='data/training.h5',
                        help='HDF5 input (training) file')
    parser.add_argument('-o', default='data/dataset.h5',
                        help='HDF5 output file')
    args = parser.parse_args()

    with h5py.File(args.i, 'r') as f_h5:
        with h5py.File(args.o, 'w') as out_f_h5:
            generate(f_h5, out_f_h5)
