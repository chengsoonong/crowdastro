"""Loads the configuration file.

Matthew Alger
The Australian National University
2016
"""

import json
import os
import pkg_resources

import numpy

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

# TODO(MatthewJA): Don't hardcode this location.
config = json.loads(pkg_resources.resource_string(
        __name__, 'crowdastro.json').decode('utf-8'))

# Normalise paths.
# TODO(MatthewJA): Also don't hardcode these.
sources = [
    'atlas_catalogue',
    'swire_cdfs_catalogue',
    'swire_elais_catalogue',
    'atlas_cdfs_image',
    'atlas_elais_image',
    'norris_coords',
    'first_images_dir',
]
for source in sources:
    config['data_sources'][source] = os.path.normpath(
            os.path.join(os.getcwd(), config['data_sources'][source]))

# Generate some helper configuration info.
config['surveys']['atlas']['click_to_fits_x'] = (
    config['surveys']['atlas']['fits_width'] /
    config['surveys']['atlas']['click_width'])
config['surveys']['atlas']['click_to_fits_y'] = (
    config['surveys']['atlas']['fits_height'] /
    config['surveys']['atlas']['click_height'])
config['surveys']['atlas']['click_to_fits'] = numpy.array(
    [config['surveys']['atlas']['click_to_fits_x'],
     config['surveys']['atlas']['click_to_fits_y']],
    dtype=float)
config['surveys']['atlas']['web_to_click_x'] = (
    config['surveys']['atlas']['click_width'] /
    config['surveys']['atlas']['web_width'])
config['surveys']['atlas']['web_to_click_y'] = (
    config['surveys']['atlas']['click_height'] /
    config['surveys']['atlas']['web_height'])
config['surveys']['atlas']['web_to_click'] = numpy.array(
    [config['surveys']['atlas']['web_to_click_x'],
     config['surveys']['atlas']['web_to_click_y']],
    dtype=float)


def get(*args, **kwargs):
    # Wrapper function to neatly retrieve values.
    return config.get(*args, **kwargs)
