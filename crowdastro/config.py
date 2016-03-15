"""Loads the configuration file."""

import json
import os.path

import numpy

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

CONFIG_PATH = os.path.join(SCRIPT_PATH, 'crowdastro.json')

with open(CONFIG_PATH) as config_file:
    config = json.load(config_file)

# Normalise paths.
config['data_path'] = os.path.normpath(config['data_path'])
config['atlas_catalogue_path'] = os.path.normpath(config['atlas_catalogue_path'])

# Generate some helper configuration info.
config['click_to_fits_x'] = (config.get('fits_image_width') /
                             config.get('click_image_width'))
config['click_to_fits_y'] = (config.get('fits_image_height') /
                             config.get('click_image_height'))
config['click_to_fits'] = numpy.array([config['click_to_fits_x'],
                                       config['click_to_fits_y']])

# Wrapper function to neatly retrieve values.
def get(*args, **kwargs):
    return config.get(*args, **kwargs)
