"""Command-line scripts for crowdastro."""

import argparse
import logging
import sys

from . import labels
from . import training_data as td
from . import catalogue as cat

def raw_classifications(args):
    """Processes raw classifications from the Radio Galaxy Zoo database."""
    labels.freeze_classifications(args.output, args.table, atlas=args.atlas)

def consensuses(args):
    """Processes consensuses from the Radio Galaxy Zoo database."""
    labels.freeze_consensuses(args.database, args.classification_table,
                              args.consensus_table, atlas=args.atlas)

def training_data(args):
    """Generates training data."""
    td.generate(args.database, args.consensus_table, args.cache, args.output,
                atlas=args.atlas)

def catalogue(args):
    """Generates the Radio Galaxy Zoo catalogue."""
    cat.generate(args.database, args.cache, args.consensus_table,
                 args.host_table, args.radio_table, atlas=args.atlas)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', '-v', action='store_true',
            help='verbose output')
    subparsers = parser.add_subparsers()

    parser_raw_classifications = subparsers.add_parser('raw_classifications',
            help='Process raw classifications from the Radio Galaxy Zoo '
                 'database.')
    parser_raw_classifications.add_argument('output',
            help='path to output SQLite database')
    parser_raw_classifications.add_argument('table', help='database table name')
    parser_raw_classifications.add_argument('--atlas', action='store_true',
            help='only process ATLAS subjects')
    parser_raw_classifications.set_defaults(func=raw_classifications)

    parser_consensuses = subparsers.add_parser('consensuses',
            help='Process consensuses from the Radio Galaxy Zoo database.')
    parser_consensuses.add_argument('database', help='path to SQLite database')
    parser_consensuses.add_argument('classification_table',
            help='name of classification database table')
    parser_consensuses.add_argument('consensus_table',
            help='name of consensus database table')
    parser_consensuses.add_argument('--atlas', action='store_true',
            help='only process ATLAS subjects')
    parser_consensuses.set_defaults(func=consensuses)

    parser_training_data = subparsers.add_parser('training_data',
            help='Generate training data.')
    parser_training_data.add_argument('database',
            help='path to SQLite database')
    parser_training_data.add_argument('consensus_table',
            help='name of consensus database table')
    parser_training_data.add_argument('cache', help='name of Gator cache')
    parser_training_data.add_argument('output', help='path to output HDF5 file')
    parser_training_data.add_argument('--atlas', action='store_true',
            help='only process ATLAS subjects')
    parser_training_data.set_defaults(func=training_data)

    parser_catalogue = subparsers.add_parser('catalogue',
            help='Generate Radio Galaxy Zoo catalogue.')
    parser_catalogue.add_argument('database',
            help='path to SQLite database')
    parser_catalogue.add_argument('consensus_table',
            help='name of consensus database table')
    parser_catalogue.add_argument('cache', help='name of Gator cache')
    parser_catalogue.add_argument('host_table',
            help='name of output host database table')
    parser_catalogue.add_argument('radio_table',
            help='name of output radio component database table')
    parser_catalogue.add_argument('--atlas', action='store_true',
            help='only process ATLAS subjects')
    parser_catalogue.set_defaults(func=catalogue)

    args = parser.parse_args()

    logging.captureWarnings(True)  # Mainly for Astropy.
    if args.verbose:
        logging.root.setLevel(logging.DEBUG)
    else:
        logging.root.setLevel(logging.INFO)
    
    if not hasattr(args, 'func'):
        parser.print_help()
    else:
        args.func(args)

if __name__ == '__main__':
    sys.exit(main())
