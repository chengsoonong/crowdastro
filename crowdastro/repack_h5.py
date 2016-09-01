#!/usr/bin/env python3
"""
Repacks an HDF5 file.

Matthew Alger
The Australian National University
2016
"""

import argparse
import os
import shutil
import tempfile

import h5py


def _populate_parser(parser):
    parser.description = 'Repacks an HDF5 file.'
    parser.add_argument('h5', help='HDF5 file to repack')


def _main(args):
    with tempfile.TemporaryDirectory() as tempdir:
        temp_h5 = os.path.join(tempdir, 'temp.h5')
        with h5py.File(args.h5, 'r') as input_h5:
            with h5py.File(temp_h5, 'w') as output_h5:
                for key in input_h5:
                    output_h5[key] = input_h5[key].value
                for key, val in input_h5.attrs.items():
                    output_h5.attrs[key] = val

        os.remove(args.h5)
        shutil.copyfile(temp_h5, args.h5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    _populate_parser(parser)
    args = parser.parse_args()
    _main(args)
