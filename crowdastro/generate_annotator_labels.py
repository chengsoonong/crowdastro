"""Generates galaxy labels for individual annotators.

Matthew Alger
The Australian National University
2016
"""

import argparse
import logging

import h5py
import numpy
import sklearn.neighbors

from .config import config

ARCMIN = 1 / 60  # deg
IMAGE_SIZE = {}
IMAGE_SIZE['atlas'] = (config['surveys']['atlas']['fits_width'] *
                       config['surveys']['atlas']['fits_height'])
IMAGE_SIZE['first'] = (config['surveys']['first']['fits_width'] *
                       config['surveys']['first']['fits_height'])


def main(f_h5, ir_survey, radio_survey, overwrite=False):
    """
    f_h5: crowdastro HDF5 file.
    ir_survey: 'wise' or 'swire'.
    radio_survey: 'first' or 'atlas'.
    overwrite: Whether to overwrite existing annotator labels. Default False.
    """
    if radio_survey == 'atlas':
        radio_prefix = '/atlas/cdfs/'
        ir_prefix = '/{}/cdfs/'.format(ir_survey)
    elif radio_survey == 'first':
        radio_prefix = '/first/first/'
        ir_prefix = '/{}/first/'.format(ir_survey)

    usernames = sorted({username for username in
                        f_h5[radio_prefix + 'classification_usernames'].value
                        if username})
    usernames = {j:i for i, j in enumerate(usernames)}
    n_annotators = len(usernames)
    n_examples = f_h5[ir_prefix + 'numeric'].shape[0]

    if ((ir_prefix + 'rgz_raw_labels' in f_h5 or
         ir_prefix + 'rgz_raw_labels_mask' in f_h5) and overwrite):
        del f_h5[ir_prefix + 'rgz_raw_labels']
        del f_h5[ir_prefix + 'rgz_raw_labels_mask']

    labels = f_h5.create_dataset(
            ir_prefix + 'rgz_raw_labels',
            data=numpy.zeros((n_annotators, n_examples)).astype(bool))
    unseen = f_h5.create_dataset(
            ir_prefix + 'rgz_raw_labels_mask',
            data=numpy.ones((n_annotators, n_examples)).astype(bool))

    ir_tree = sklearn.neighbors.KDTree(f_h5[ir_prefix + 'numeric'][:, :2])

    # What objects has each annotator seen? Assume that if they have labelled a
    # radio object, then they have seen everything within 1 arcminute.
    for class_pos, username in zip(
            f_h5[radio_prefix + 'classification_positions'],
            f_h5[radio_prefix + 'classification_usernames']):
        if username not in usernames:
            # Skip unidentified users.
            continue

        radio_id, ra, dec, primary = class_pos
        if radio_survey == 'atlas':
            distances = f_h5[radio_prefix + 'numeric'][
                radio_id, 2 + IMAGE_SIZE['atlas']:]
        elif radio_survey == 'first':
            # Images aren't included for FIRST.
            distances = f_h5[radio_prefix + 'numeric'][radio_id, 2:]
        else:
            raise NotImplementedError()

        assert distances.shape[0] == n_examples
        nearby = distances <= ARCMIN
        unseen[usernames[username], nearby] = False  # Unmask nearby objects.

        # Now, label the example we are looking at now.
        if numpy.isnan(ra) or numpy.isnan(dec):
            # No IR source, so we can just skip this.
            continue

        # Get the nearest IR object and assign the label.
        ((dist,),), ((ir,),) = ir_tree.query([(ra, dec)])
        if dist > config['surveys'][ir_survey]['distance_cutoff']:
            logging.debug('Click has no corresponding IR.')
            continue

        labels[usernames[username], ir] = 1


def _populate_parser(parser):
    parser.description = 'Generates galaxy labels for individual annotators.'
    parser.add_argument('--h5', default='data/crowdastro.h5',
                        help='HDF5 IO file')
    parser.add_argument('--overwrite', action='store_true',
                        help='overwrite existing annotator labels')
    parser.add_argument('--survey', default='atlas', choices=['atlas', 'first'],
                        help='Radio survey to generate consensuses for.')


def _main(args):
    with h5py.File(args.h5, 'r+') as f_h5:
        ir_survey = f_h5.attrs['ir_survey']
        main(f_h5, ir_survey, args.survey, overwrite=args.overwrite)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    _populate_parser(parser)
    args = parser.parse_args()
    _main(args)
