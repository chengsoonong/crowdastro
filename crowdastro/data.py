"""Utilities for interacting with the Radio Galaxy Zoo data."""

import os.path

import astropy.io.fits
import matplotlib.colors
import matplotlib.pyplot
import numpy
import pymongo
import requests

from . import config

client = pymongo.MongoClient(config.get('host'), config.get('port'))
db = client[config.get('db_name')]

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
