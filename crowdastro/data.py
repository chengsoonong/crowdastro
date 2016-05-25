"""Utilities for interacting with the Radio Galaxy Zoo data."""

import io
import os.path

import astropy.io.fits
import astropy.io.votable
import astropy.wcs
import matplotlib.colors
import matplotlib.pyplot
import numpy
import pymongo
import requests
import requests_cache

from . import config

client = pymongo.MongoClient(config.get('mongo_host'), config.get('mongo_port'))
db = client[config.get('mongo_db_name')]

def require_atlas(f):
    """Decorator that ensures a subject (the first argument) is from the ATLAS
    survey.
    """
    def g(subject, *args, **kwargs):
        if subject['metadata']['survey'] != 'atlas':
            raise ValueError('Subject not from ATLAS survey.')

        return f(subject, *args, **kwargs)

    return g

def get_random_subject():
    return list(db.radio_subjects.aggregate([
        {'$match': {'metadata.survey': 'atlas'}},
        {'$sample': {'size': 1}}]))[0]

def get_subject(zid):
    """Gets a Radio Galaxy Zoo subject from the database.

    zid: Zooniverse ID of subject.
    -> RGZ subject dict.
    """
    return db.radio_subjects.find_one({'zooniverse_id': zid})

def get_subject_raw_id(subject_id):
    """Gets a Radio Galaxy Zoo subject from the database using its subject ID.

    subject_id: Raw ID of subject.
    -> RGZ subject dict.
    """
    return db.radio_subjects.find_one({'_id': subject_id})

def get_all_classifications():
    """Yields all RGZ classification dicts."""
    return db.radio_classifications.find()

def get_subject_classifications(subject):
    """Yields all classifications associated with a subject.

    subject: RGZ subject dict.
    """
    return db.radio_classifications.find({'subject_ids': subject['_id']})

def get_all_subjects(atlas=False):
    """Yields all RGZ subject dicts.

    atlas: Whether to only yield ATLAS subjects. Default False.
    """
    if atlas:
        return db.radio_subjects.find(
                {'metadata.survey': 'atlas'}).batch_size(100)

    return db.radio_subjects.find().batch_size(100)

@require_atlas
def open_fits(subject, field, wavelength, size='2x2'):
    """Opens a FITS image of a subject.

    Can be used as a context handler.

    subject: RGZ subject dict, from the ATLAS survey.
    field: 'elais' or 'cdfs'
    wavelength: 'ir' or 'radio'
    size: Optional. '2x2' or '5x5'.
    -> FITS image file.
    """
    if field not in {'elais', 'cdfs'}:
        raise ValueError('field must be either "elais" or "cdfs".')

    if wavelength not in {'ir', 'radio'}:
        raise ValueError('wavelength must be either "ir" or "radio".')

    cid = subject['metadata']['source']
    filename = '{}_{}.fits'.format(cid, wavelength)
    path = os.path.join(config.get('data_path'), field, size, filename)
    
    return astropy.io.fits.open(path, ignore_blank=True)

@require_atlas
def get_ir(subject, size='2x2'):
    """Returns the IR image of a subject.

    subject: RGZ subject dict, from the ATLAS survey.
    size: Optional. '2x2' or '5x5'.
    -> NumPy array.
    """
    if subject['metadata']['source'].startswith('C'):
        field = 'cdfs'
    else:
        field = 'elais'

    with open_fits(subject, field, 'ir', size=size) as fits_file:
        return fits_file[0].data

@require_atlas
def get_ir_fits(subject, size='2x2'):
    """Returns the IR image of a subject as a FITS image.

    subject: RGZ subject dict, from the ATLAS survey.
    size: Optional. '2x2' or '5x5'.
    -> FITS image.
    """
    if subject['metadata']['source'].startswith('C'):
        field = 'cdfs'
    else:
        field = 'elais'

    with open_fits(subject, field, 'ir', size=size) as fits_file:
        return fits_file[0]

@require_atlas
def get_radio(subject, size='2x2'):
    """Returns the radio image of a subject.

    subject: RGZ subject dict, from the ATLAS survey.
    size: Optional. '2x2' or '5x5'.
    -> NumPy array.
    """
    if subject['metadata']['source'].startswith('C'):
        field = 'cdfs'
    else:
        field = 'elais'

    with open_fits(subject, field, 'radio', size=size) as fits_file:
        return fits_file[0].data

@require_atlas
def get_radio_fits(subject, size='2x2'):
    """Returns the radio image of a subject as a FITS image.

    subject: RGZ subject dict, from the ATLAS survey.
    size: Optional. '2x2' or '5x5'.
    -> FITS image.
    """
    if subject['metadata']['source'].startswith('C'):
        field = 'cdfs'
    else:
        field = 'elais'

    with open_fits(subject, field, 'radio', size=size) as fits_file:
        return fits_file[0]

def get_contours(subject):
    """Fetches the radio contours of a subject.

    subject: RGZ subject dict.
    -> JSON dict.
    """
    # TODO(MatthewJA): Cache these.
    return requests.get(subject['location']['contours']).json()

@require_atlas
def get_potential_hosts(subject, cache_name, convert_to_px=True):
    """Finds the potential hosts for a subject.

    subject: RGZ subject dict.
    cache_name: Name of Gator cache.
    convert_to_px: Whether to convert coordinates to pixels. Default True; if
        False then coordinates will be RA/DEC.
    -> dict mapping (x, y) tuples to
        - flux at 3.6μm for aperture #2
        - flux at 4.5μm for aperture #2
        - flux at 5.8μm for aperture #2
        - flux at 8.0μm for aperture #2
        - flux at 24μm for aperture #2
        - stellarity index at 3.6μm
        - uncertainty in RA
        - uncertainty in DEC
    """
    requests_cache.install_cache(cache_name=cache_name, backend='sqlite',
                                 expire_after=None)

    if subject['metadata']['source'].startswith('C'):
        # CDFS
        catalog = 'chandra_cat_f05'
    else:
        # ELAIS-S1
        catalog = 'elaiss1_cat_f05'
    
    query = {
        'catalog': catalog,
        'spatial': 'box',
        'objstr': '{} {}'.format(*subject['coords']),
        'size': '120',
        'outfmt': '3',
    }
    url = 'http://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query'

    r = requests.get(url, params=query)
    votable = astropy.io.votable.parse_single_table(io.BytesIO(r.content),
                                                    pedantic=False)
    
    ras = votable.array['ra']
    decs = votable.array['dec']

    if convert_to_px:
        # Convert to px.
        fits = get_ir_fits(subject)
        wcs = astropy.wcs.WCS(fits.header)
        xs, ys = wcs.all_world2pix(ras, decs, 0)
    else:
        xs, ys = ras, decs
    
    # Get the astronomical features.
    out = {}  # Maps (x, y) to astronomical features.
    for x, y, row_idx in zip(xs, ys, range(votable.nrows)):
        row = votable.array[row_idx]
        out[x, y] = {
            'name': row['object'],
            'clon': row['clon'],
            'clat': row['clat'],
            'flux_ap2_36': row['flux_ap2_36'],
            'flux_ap2_45': row['flux_ap2_45'],
            'flux_ap2_58': row['flux_ap2_58'],
            'flux_ap2_80': row['flux_ap2_80'],
            'flux_ap2_24': row['flux_ap2_24'],
            'stell_36': row['stell_36'],
            'unc_ra': row['unc_ra'],
            'unc_dec': row['unc_dec'],
        }

    requests_cache.uninstall_cache()

    return out
