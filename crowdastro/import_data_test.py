"""Tests for import_data.py.

Matthew Alger
The Australian National University
2016
"""

#/usr/bin/env python3

import shutil
import tempfile
import unittest

import h5py
import numpy

from . import config
from . import import_data


# Most of these tests are just regression-based sanity checks. This should help
# catch almost all bugs we've had so far, though.

class TestImportAtlas(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.f_h5 = h5py.File('test.h5', 'w')
        self.f_csv = open('test.csv', 'w')
        import_data.config['data_sources']['rgz_to_atlas'] = (
                config.config['data_sources']['test_rgz_to_atlas'])
        import_data.prep_h5(self.f_h5)

    def tearDown(self):
        self.f_h5.close()
        self.f_csv.close()
        shutil.rmtree(self.test_dir)

    def test_coords(self):
        """Coordinates are imported into the HDF5 file."""
        import_data.import_atlas(self.f_h5, self.f_csv)
        numpy.allclose(
                self.f_h5['/atlas/cdfs/positions'].value,
                [[ 51.89172 , -28.772626],
                 [ 53.538672, -28.405543],
                 [ 52.853988, -28.303602],
                 [ 52.194519, -28.43758 ],
                 [ 53.972284, -27.461299],
                 [ 53.864673, -27.330802],
                 [ 52.317969, -27.394784]])

    # TODO(MatthewJA): Write some actual tests.


if __name__ == '__main__':
    unittest.main()
