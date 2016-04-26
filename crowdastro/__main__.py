"""Command-line scripts for crowdastro."""

import argparse
import sys

from . import labels

def raw_classifications(args):
    """Processes raw classifications from the Radio Galaxy Zoo database."""
    labels.freeze_classifications(args.output, args.table)

def consensuses(args):
    """Processes consensuses from the Radio Galaxy Zoo database."""
    labels.freeze_consensuses(args.database, args.classification_table,
                              args.consensus_table, atlas=args.atlas)

def training_data(args):
    """Generates training data."""
    training_data.generate(args.database, args.consensus_table, args.model,
                           args.weights, args.cache, args.output)

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_raw_classifications = subparsers.add_parser('raw_classifications',
            help='Process raw classifications from the Radio Galaxy Zoo '
                 'database.')
    parser_raw_classifications.add_argument('output',
            help='path to output SQLite database')
    parser_raw_classifications.add_argument('table', help='database table name')
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
    parser_training_data.add_argument('model', help='path to CNN model')
    parser_training_data.add_argument('weights', help='path to CNN weights')
    parser_training_data.add_argument('cache', help='name of Gator cache')
    parser_training_data.add_argument('output', help='path to output HDF5 file')

    args = parser.parse_args()
    
    if not hasattr(args, 'func'):
        parser.print_help()
    else:
        args.func(args)

if __name__ == '__main__':
    sys.exit(main())
