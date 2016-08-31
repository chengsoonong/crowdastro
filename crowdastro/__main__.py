"""Command-line interface for crowdastro.

Matthew Alger
The Australian National University
2016
"""

import argparse
import logging
import sys

from . import __description__
from . import __version__
from . import compile_cnn
from . import consensuses
from . import generate_cnn_outputs
from . import generate_dataset
from . import generate_test_sets
from . import generate_training_data
from . import import_data
from . import repack_h5
from . import test
from . import train
from . import train_cnn

def main():
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument('--verbose', '--v', '-v', action='store_true',
            help='verbose output')
    parser.add_argument('--version', action='store_true',
            help='get version number')
    subparsers = parser.add_subparsers(dest='subcommand')

    parser_compile_cnn = subparsers.add_parser('compile_cnn',
            help='compile a convolutional neural network')
    compile_cnn._populate_parser(parser_compile_cnn)

    parser_consensuses = subparsers.add_parser('consensuses',
            help='generate Radio Galaxy Zoo consensus classifications')
    consensuses._populate_parser(parser_consensuses)

    parser_generate_cnn_outputs = subparsers.add_parser('generate_cnn_outputs',
            help='generate convolutional neural network training outputs')
    generate_cnn_outputs._populate_parser(parser_generate_cnn_outputs)

    parser_generate_dataset = subparsers.add_parser('generate_dataset',
            help='generate crowdastro dataset')
    generate_dataset._populate_parser(parser_generate_dataset)

    parser_generate_test_sets = subparsers.add_parser('generate_test_sets',
            help='generate crowdastro galaxy test sets')
    generate_test_sets._populate_parser(parser_generate_test_sets)

    parser_generate_training_data = subparsers.add_parser(
            'generate_training_data',
            help='generate crowdastro galaxy test sets')
    generate_training_data._populate_parser(parser_generate_training_data)

    parser_import_data = subparsers.add_parser('import_data',
            help='import data into crowdastro')
    import_data._populate_parser(parser_import_data)

    parser_repack_h5 = subparsers.add_parser('repack_h5',
            help='repacks an HDF5 file')
    repack_h5._populate_parser(parser_repack_h5)

    parser_test = subparsers.add_parser('test',
            help='tests classifiers')
    test._populate_parser(parser_test)

    parser_train = subparsers.add_parser('train',
            help='trains classifiers')
    train._populate_parser(parser_train)

    parser_train_cnn = subparsers.add_parser('train_cnn',
            help='trains a convolutional neural network')
    train_cnn._populate_parser(parser_train_cnn)

    # http://stackoverflow.com/a/11287731/1105803
    if len(sys.argv) < 2:
        sys.argv.append('--help')

    args = parser.parse_args()

    logging.captureWarnings(True)
    if args.verbose:
        logging.root.setLevel(logging.DEBUG)

    if args.version:
        print(__version__)
        return

    subcommands = {
        'compile_cnn': compile_cnn._main,
        'consensuses': consensuses._main,
        'generate_cnn_outputs': generate_cnn_outputs._main,
        'generate_dataset': generate_dataset._main,
        'generate_test_sets': generate_test_sets._main,
        'generate_training_data': generate_training_data._main,
        'import_data': import_data._main,
        'repack_h5': repack_h5._main,
        'test': test._main,
        'train': train._main,
        'train_cnn': train_cnn._main,
    }

    subcommands[args.subcommand](args)


if __name__ == '__main__':
    main()
