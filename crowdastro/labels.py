"""Utilities for manipulating labels from the Radio Galaxy Zoo."""

import functools
import operator
import sqlite3
import struct

from . import config
from . import data

DEFAULT_SCALE_WIDTH = 2.1144278606965172
FP_PRECISION = 14

def make_radio_combination_signature(radio_annotation):
    """Generates a unique signature for a radio annotation.
    
    radio_annotation: 'radio' dictionary from a classification.
    -> Something immutable
    """
    # My choice of immutable object will be a bytes object. This will be an
    # encoding of a tuple of the xmax values, sorted to ensure determinism, and
    # rounded to nix floating point errors.
    xmaxes = []
    for c in radio_annotation.values():
        # Note that the x scale is not the same as the IR scale, but the scale
        # factor is included in the annotation, so I have multiplied this out
        # here for consistency.
        scale_width = c.get('scale_width', '')
        if scale_width:
            scale_width = float(scale_width)
        else:
            # Sometimes, there's no scale, so I've included a default scale.
            scale_width = DEFAULT_SCALE_WIDTH

        # Keep everything within the maximum width, so that we can accurately
        # guess precision.
        scale_width /= config.get('click_image_width')

        xmax = round(float(c['xmax']) * scale_width, FP_PRECISION)

        # Some of the numbers are out of range. I don't have a good fallback, so
        # I'll just raise an error.
        if not 0 <= xmax <= 1:
            raise ValueError('Radio xmax out of range: {}'.format(xmax))
        
        xmaxes.append(xmax)

    xmaxes.sort()

    # Pack the floats to convert them into bytes. I will use little-endian,
    # 32-bit floats.
    xmaxes = [struct.pack('<f', xmax) for xmax in xmaxes]

    return functools.reduce(operator.add, xmaxes, b'')

def parse_classification(classification):
    """Converts a raw RGZ classification into a classification dict.

    classification: RGZ classification dict.
    -> (subject MongoDB ID,
        dict mapping radio signature to corresponding IR host pixel location)
    """
    result = {}

    for annotation in classification['annotations']:
        if 'radio' not in annotation:
            # This is a metadata annotation and we can ignore it.
            continue

        if annotation['radio'] == 'No Contours':
            # I'm not sure how this occurs. I'm going to ignore it.
            continue

        try:
            radio_signature = make_radio_combination_signature(
                    annotation['radio'])
        except ValueError:
            # Ignore invalid annotations.
            continue

        if annotation['ir'] == 'No Sources':
            ir_location = None
        else:
            ir_x = float(annotation['ir']['0']['x'])
            ir_y = float(annotation['ir']['0']['y'])

            # Ignore out-of-range data.
            if not 0 <= ir_x <= config.get('click_image_width'):
                continue

            if not 0 <= ir_y <= config.get('click_image_height'):
                continue

            ir_location = (ir_x, ir_y)

        result[radio_signature] = ir_location

    assert len(classification['subject_ids']) == 1

    return classification['subject_ids'][0], result

def freeze_classifications(db_path, table):
    """Freezes Radio Galaxy Zoo classifications into a SQLite database.

    Warning: table argument is not validated! This could be dangerous.

    db_path: Path to SQLite database. If this doesn't exist, it will be created.
    table: Name of table to freeze classifications into. If this exists, it will
        be cleared.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute('DROP TABLE IF EXISTS {}'.format(table))
    conn.commit()

    c.execute('CREATE TABLE {} (subject_id TEXT, radio_signature BLOB, '
              'source_x REAL, source_y REAL)'.format(table))
    conn.commit()

    sql = ('INSERT INTO {} (subject_id, radio_signature, source_x, '
           'source_y) VALUES (?, ?, ?, ?)'.format(table))

    def iter_sql_params():
        for c in data.get_all_classifications():
            subject_id, radio_locations = parse_classification(c)
            for radio, location in radio_locations.items():
                if location is None:
                    x = None
                    y = None
                else:
                    x, y = location

                yield (str(subject_id), radio, x, y)

    c.executemany(sql, iter_sql_params())

    conn.commit()
