"""Generates the Radio Galaxy Zoo catalogue."""

import contextlib
import logging
import sqlite3

import astropy.coordinates
import astropy.units
import astropy.wcs
import numpy

from . import data

HOURS_PER_DEGREE = 1 / 15

def distance_from(point):
    """Returns a function that returns the distance from a point.

    Used for keys.
    """
    def _dist(other):
        return numpy.hypot(point[0] - other[0], point[1] - other[1])
    return _dist

def make_host(subject, wcs, cache_name, consensus):
    """Returns an RGZ and SWIRE name for the host, and the location.

    subject: RGZ subject dict.
    wcs: World coordinate system (from Astropy).
    cache_name: Name of Gator cache.
    consensus: Consensus dict from database.
    -> (str: RGZ_JHHMMSS-DDMMSS, SWIRE name, RA, DEC)
    """
    # We want RGZ_JHHMMSS-DDMMSS and an associated SWIRE result. Let's start
    # with the SWIRE result so that the coordinates we get are accurate and
    # reproducible. Convert pixel coordinates into RA/DEC.
    x, y = consensus['source_x'], consensus['source_y']
    x, y = wcs.all_pix2world([x], [y], 0)

    # Get the closest SWIRE object.
    p_hosts = data.get_potential_hosts(subject, cache_name,
                                       convert_to_px=False)
    dist_func = distance_from((x, y))
    nearest = min(p_hosts, key=dist_func)
    swire_name = p_hosts[nearest]['name'].decode('ascii')

    # This provides our coordinates. Separate out HHMMSS/DDMMSS. We can do this
    # using astropy.coordinates.SkyCoord. Note that I'm using the clon/clat
    # values returned by Gator; this helps keep us lined up with the SWIRE
    # coordinates since if you use the RA/DEC raw you get rounding errors.
    clon = p_hosts[nearest]['clon'].decode('ascii')
    clat = p_hosts[nearest]['clat'].decode('ascii')
    coords = astropy.coordinates.SkyCoord(clon + ' ' + clat, unit='deg')
    ra = coords.ra.hms
    dec = coords.dec.signed_dms

    # Construct the name.
    rgz_name = ('RGZ_J{ra_h:02}{ra_m:02}{ra_s:02}{sign}'
                '{dec_d:02}{dec_m:02}{dec_s:02}').format(
                ra_h=int(ra[0]),
                ra_m=int(ra[1]),
                ra_s=int(round(ra[2])),
                sign='-' if dec[0] < 0 else '+',
                dec_d=int(dec[1]),
                dec_m=int(dec[2]),
                dec_s=int(round(dec[3])))

    return rgz_name, swire_name, clon, clat

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


        cur.execute('DROP TABLE IF EXISTS {}'.format(host_table))
        cur.execute('DROP TABLE IF EXISTS {}'.format(radio_table))
        conn.commit()

        cur.execute('CREATE TABLE {} '
                    '(zooniverse_id TEXT, source TEXT, rgz_name TEXT, '
                    'swire_name TEXT, ra TEXT, dec TEXT)'.format(host_table))
        cur.execute('CREATE TABLE {} '
                    '(rgz_name TEXT, radio_component TEXT)'.format(radio_table))
        conn.commit()

        host_sql = ('INSERT INTO {} (zooniverse_id, source, rgz_name, '
                    'swire_name, ra, dec) VALUES '
                    '(?, ?, ?, ?, ?, ?)'.format(host_table))
        radio_sql = ('INSERT INTO {} (rgz_name, radio_component) VALUES '
                     '(?, ?)'.format(radio_table))

        host_params = []
        radio_params = []

        n_subjects = data.get_all_subjects(atlas=atlas).count()
        for index, subject in enumerate(data.get_all_subjects(atlas=atlas)):
            print('Generating catalogue: {}/{} ({:.02%})'.format(
                    index + 1, n_subjects, (index + 1) / n_subjects), end='\r')
            consensuses = cur.execute(
                    'SELECT * FROM {} WHERE '
                    'zooniverse_id = ?'.format(consensus_table),
                    [subject['zooniverse_id']])

            fits = data.get_ir_fits(subject)
            wcs = astropy.wcs.WCS(fits.header)

            for consensus in consensuses:
                # Each consensus represents one AGN.
                if consensus['source_x'] and consensus['source_y']:
                    # Not null.
                    rgz_name, swire_name, ra, dec = make_host(subject, wcs,
                                                              cache_name,
                                                              consensus)
                    host_params.append((subject['zooniverse_id'],
                                        subject.get('metadata', {}).get(
                                                'source'),
                                        rgz_name, swire_name, ra, dec))

                    # Get radio components.
                    radio_components = set(  # Set to nix duplicates.
                            consensus['radio_signature'].split(';'))
                    for radio_component in radio_components:
                        radio_params.append((rgz_name, radio_component))
                else:
                    logging.debug('Skipping null consensus for subject %s.',
                                  subject['zooniverse_id'])

        logging.debug('Writing to database.')
        cur.executemany(host_sql, host_params)
        cur.executemany(radio_sql, radio_params)
        conn.commit()
