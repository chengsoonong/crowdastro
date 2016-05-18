"""Generates training data (potential hosts and their astronomical features)."""

import contextlib
import sqlite3

import keras.models
import numpy
import pandas

from . import config
from . import data
from . import labels

RADIO_PADDING = (500 - 200) // 2
PATCH_RADIUS = 40  # 80 x 80 patches.

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
        output_ids = []
        output_sources = []
        output_xs = []
        output_ys = []
        output_flux_ap2_24 = []
        output_flux_ap2_36 = []
        output_flux_ap2_45 = []
        output_flux_ap2_58 = []
        output_flux_ap2_80 = []
        output_labels = []

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
                output_ids.append(subject['zooniverse_id']),
                output_sources.append(subject['metadata']['source']),
                output_xs.append(host_x)
                output_ys.append(host_y)
                output_labels.append(astro['is_host'])
                output_flux_ap2_24.append(astro['flux_ap2_24'])
                output_flux_ap2_36.append(astro['flux_ap2_36'])
                output_flux_ap2_45.append(astro['flux_ap2_45'])
                output_flux_ap2_58.append(astro['flux_ap2_58'])
                output_flux_ap2_80.append(astro['flux_ap2_80'])

        output_ids = pandas.DataFrame(output_ids, dtype='S24')
        output_xs = pandas.DataFrame(output_xs, dtype=float)
        output_ys = pandas.DataFrame(output_ys, dtype=float)
        output_labels = pandas.DataFrame(output_labels, dtype=float)
        output_flux_ap2_24 = pandas.DataFrame(output_flux_ap2_24, dtype=float)
        output_flux_ap2_36 = pandas.DataFrame(output_flux_ap2_36, dtype=float)
        output_flux_ap2_45 = pandas.DataFrame(output_flux_ap2_45, dtype=float)
        output_flux_ap2_58 = pandas.DataFrame(output_flux_ap2_58, dtype=float)
        output_flux_ap2_80 = pandas.DataFrame(output_flux_ap2_80, dtype=float)

        frame = pandas.concat(
            [output_ids, output_source, output_xs, output_ys,
             output_flux_ap2_24, output_flux_ap2_36, output_flux_ap2_45,
             output_flux_ap2_58, output_flux_ap2_80, output_labels], axis=1,
            keys=['zooniverse_id', 'source', 'x', 'y', 'flux_ap2_24',
                  'flux_ap2_36', 'flux_ap2_45', 'flux_ap2_58', 'flux_ap2_80',
                  'is_host'])

        with pandas.HDFStore(output_path) as store:
            store['data'] = frame
