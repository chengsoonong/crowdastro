#!/usr/bin/env python3
"""
Repacks an H5 file.

Matthew Alger
The Australian National University
2016
"""

import argparse
import os
import shutil
import tempfile

import h5py

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('h5', help='HDF5 file to repack')
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tempdir:
        temp_h5 = os.path.join(tempdir, 'temp.h5')
        with h5py.File(args.h5, 'r') as input_h5:
            with h5py.File(temp_h5, 'w') as output_h5:
                for key in input_h5:
                    output_h5[key] = input_h5[key].value

        os.remove(args.h5)
        shutil.copyfile(temp_h5, args.h5)