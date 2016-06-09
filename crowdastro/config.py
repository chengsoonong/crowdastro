"""Loads the configuration file.

Matthew Alger
The Australian National University
2016
"""

import json
import os.path

import numpy

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

# TODO(MatthewJA): Don't hardcode this.
CONFIG_PATH = os.path.join(SCRIPT_PATH, '../crowdastro.json')

with open(CONFIG_PATH) as config_file:
    config = json.load(config_file)

# Normalise paths.
# TODO(MatthewJA): Also don't hardcode these. They should be relative to the
# JSON(?).
config['data_sources']['atlas_catalogue'] = os.path.normpath(
        os.path.join(SCRIPT_PATH, '..',
                     config['data_sources']['atlas_catalogue']))
config['data_sources']['cdfs_fits'] = os.path.normpath(
        os.path.join(SCRIPT_PATH, '..',
                     config['data_sources']['cdfs_fits']))
config['data_sources']['swire_catalogue'] = os.path.normpath(
        os.path.join(SCRIPT_PATH, '..',
                     config['data_sources']['swire_catalogue']))

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

# Wrapper function to neatly retrieve values.
def get(*args, **kwargs):
    return config.get(*args, **kwargs)
