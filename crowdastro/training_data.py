"""Generates training data (potential hosts and their astronomical features)."""

import contextlib
import sqlite3

import h5py
import keras.models
import numpy

from . import config
from . import data
from . import labels

RADIO_PADDING = (500 - 200) // 2
PATCH_RADIUS = 40  # 80 x 80 patches.

def remove_nans(n):
    """Replaces NaN with 0."""
    if numpy.ma.is_masked(n):
        return 0
    return float(n)

def generate(db_path, consensus_table, cache_name, output_path, atlas=False):
    """Generates potential hosts and their astronomical features.

    db_path: Path to consensus SQLite database.
    consensus_table: Name of the consensus table in the database.
    cache_name: Name of Gator cache.
    output_path: Path to output HDF5 file. Training data will be output here.
    atlas: Optional. Whether to only use ATLAS data.
    """
    with contextlib.closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # We'll store our output here, then dump to HDF5.
        output = []

        n_subjects = data.get_all_subjects(atlas=atlas).count()
        for index, subject in enumerate(data.get_all_subjects(atlas=atlas)):
            print('Generating training data: {}/{} ({:.02%})'.format(
                    index + 1, n_subjects, (index + 1) / n_subjects), end='\r')
            consensuses = cur.execute(
                    'SELECT * FROM {} WHERE '
                    'zooniverse_id = ?'.format(consensus_table),
                    [subject['zooniverse_id']])

            true_hosts = set()  # Set of (x, y) tuples of true hosts.
            potential_hosts = data.get_potential_hosts(subject, cache_name)

            for host in potential_hosts:
                potential_hosts[host]['is_host'] = 0

            for row in consensuses:
                cx = row['source_x']
                cy = row['source_y']
                if cx is None or cy is None:
                    continue

                closest = min(potential_hosts,
                        key=lambda z: numpy.hypot(cx - z[0], cy - z[1]))
                potential_hosts[closest]['is_host'] = 1

            # We now have a dict mapping potential hosts to astronomical
            # features.
            for (host_x, host_y), astro in potential_hosts.items():
                output.append((
                    subject['zooniverse_id'],  # S24
                    subject['metadata']['source'],  # S24
                    host_x,  # float
                    host_y,  # float
                    remove_nans(astro['flux_ap2_24']),  # float
                    remove_nans(astro['flux_ap2_36']),  # float
                    remove_nans(astro['flux_ap2_45']),  # float
                    remove_nans(astro['flux_ap2_58']),  # float
                    remove_nans(astro['flux_ap2_80']),  # float
                    astro['is_host'],  # bool
                ))

        # Pandas really, really doesn't like this data, so I'm using structured
        # NumPy arrays and h5py.
        dtype = [
            ('zooniverse_id', 'S24'),
            ('source', 'S24'),
            ('x', 'float32'),
            ('y', 'float32'),
            ('flux_ap2_24', 'float32'),
            ('flux_ap2_36', 'float32'),
            ('flux_ap2_45', 'float32'),
            ('flux_ap2_58', 'float32'),
            ('flux_ap2_80', 'float32'),
            ('is_host', 'bool'),
        ]
        struct = numpy.array(output, dtype=dtype)

        with h5py.File(output_path, 'w') as f:
            dset = f.create_dataset('data', data=struct, dtype=dtype)
