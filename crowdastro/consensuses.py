"""Finds the consensus crowd classifications in Radio Galaxy Zoo.

Matthew Alger
The Australian National University
2016
"""

def freeze_consensuses(f_h5):
    """Find Radio Galaxy Zoo crowd consensuses.

    f_h5: crowdastro HDF5 file.
    """

    c.execute('DROP TABLE IF EXISTS {}'.format(consensus_table))
    conn.commit()

    c.execute('CREATE TABLE {} (zooniverse_id TEXT, radio_signature TEXT, '
              'source_x REAL, source_y REAL, radio_agreement REAL, '
              'location_agreement REAL)'.format(consensus_table))
    conn.commit()

    sql = ('INSERT INTO {} (zooniverse_id, radio_signature, source_x, '
           'source_y, radio_agreement, location_agreement) '
           'VALUES (?, ?, ?, ?, ?, ?)'.format(consensus_table))

    params = []

    n_subjects = data.get_all_subjects(atlas=atlas).count()
    for idx, subject in enumerate(data.get_all_subjects(atlas=atlas)):
        if idx % 500 == 0:
            logging.debug('Executing %d queries.', len(params))
            c.executemany(sql, params)
            conn.commit()
            params = []

        print('Freezing consensus. Progress: {} ({:.02%})'.format(
                idx, idx / n_subjects), file=sys.stderr, end='\r')

        zooniverse_id = subject['zooniverse_id']
        cons, radio_agreement, location_agreement = (
                get_subject_consensus_kde(subject, conn, consensus_table))
        for radio_signature, (x, y) in cons.items():
            params.append((zooniverse_id, radio_signature, x, y,
                           radio_agreement,
                           location_agreement[radio_signature]))
    print()

    c.executemany(sql, params)
    conn.commit()
