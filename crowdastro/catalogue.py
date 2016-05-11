"""Generates the Radio Galaxy Zoo catalogue."""

import contextlib
import sqlite3

import astropy.wcs

from . import data

def generate(db_path, cache_name, consensus_table, host_table, radio_table,
             atlas=False):
    """Generates the Radio Galaxy Zoo catalogue.

    Warning: table arguments are not validated! This could be dangerous.

    db_path: Path to consensus database.
    cache_name: Name of Gator cache.
    consensus_table: Database table of consensuses.
    host_table: Output database table of RGZ hosts. Will be overwritten!
    radio_table: Output database table of RGZ radio sources. Will be
        overwritten!
    atlas: Whether to only freeze ATLAS subjects. Default False (though this
        function currently only works for True).
    """
    with contextlib.closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        n_subjects = data.get_all_subjects(atlas=atlas).count()
        for index, subject in enumerate(data.get_all_subjects(atlas=atlas)):
            print('Generating catalogue: {}/{} ({:.02%})'.format(
                    index + 1, n_subjects, (index + 1) / n_subjects), end='\r')
            consensuses = cur.execute(
                    'SELECT * FROM {} WHERE '
                    'subject_id = ?'.format(consensus_table),
                    [str(subject['_id'])])

            fits = data.get_ir_fits(subject)
            wcs = astropy.wcs.WCS(fits.header)
            print(consensuses)
            break