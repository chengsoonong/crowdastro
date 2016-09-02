"""Generates galaxy labels for individual annotators.

Matthew Alger
The Australian National University
2016
"""

import argparse

import h5py
import numpy
import sklearn.neighbors

from .config import config

ARCMIN = 1 / 60  # deg
ATLAS_IMAGE_SIZE = (config['surveys']['atlas']['fits_width'] * 
                    config['surveys']['atlas']['fits_height'])


def main(f_h5, ir_survey, overwrite=False):
    """
    f_h5: crowdastro HDF5 file.
    ir_survey: 'wise' or 'swire'.
    overwrite: Whether to overwrite existing annotator labels. Default False.
    """
    usernames = sorted({username for username in
                        f_h5['/atlas/cdfs/classification_usernames'].value
                        if username})
    usernames = {j:i for i, j in enumerate(usernames)}
    n_annotators = len(usernames)
    n_examples = f_h5['/{}/cdfs/numeric'.format(ir_survey)].shape[0]

    if (('/{}/cdfs/rgz_raw_labels'.format(ir_survey) in f_h5
         or '/{}/cdfs/rgz_raw_labels_mask'.format(ir_survey) in f_h5)
         and overwrite):
        del f_h5['/{}/cdfs/rgz_raw_labels'.format(ir_survey)]
        del f_h5['/{}/cdfs/rgz_raw_labels_mask'.format(ir_survey)]

    labels = f_h5.create_dataset(
            '/{}/cdfs/rgz_raw_labels'.format(ir_survey),
            data=numpy.zeros((n_annotators, n_examples)).astype(bool))
    unseen = f_h5.create_dataset(
            '/{}/cdfs/rgz_raw_labels_mask'.format(ir_survey),
            data=numpy.ones((n_annotators, n_examples)).astype(bool))

    ir_tree = sklearn.neighbors.KDTree(
            f_h5['/{}/cdfs/numeric'.format(ir_survey)][:, :2])

    # What objects has each annotator seen? Assume that if they have labelled a
    # radio object, then they have seen everything within 1 arcminute.
    for class_pos, username in zip(
            f_h5['/atlas/cdfs/classification_positions'],
            f_h5['/atlas/cdfs/classification_usernames']):
        if username not in usernames:
            # Skip unidentified users.
            continue

        atlas_id, ra, dec, primary = class_pos
        distances = f_h5['/atlas/cdfs/numeric'][atlas_id, 2 + ATLAS_IMAGE_SIZE:]
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


def _main(args):
    with h5py.File(args.h5, 'r+') as f_h5:
        ir_survey = f_h5.attrs['ir_survey']
        main(f_h5, ir_survey, overwrite=args.overwrite)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    _populate_parser(parser)
    args = parser.parse_args()
    _main(args)
