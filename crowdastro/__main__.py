"""Command-line interface for crowdastro.

Matthew Alger
The Australian National University
2016
"""

import argparse
import logging
import sys

from . import __description__
from . import compile_cnn
from . import consensuses
from . import generate_cnn_outputs
from . import generate_dataset

def main():
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument('--verbose', '-v', action='store_true',
            help='verbose output')
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

    # http://stackoverflow.com/a/11287731/1105803
    if len(sys.argv) < 2:
        sys.argv.append('--help')

    args = parser.parse_args()

    if args.verbose:
        logging.root.setLevel(logging.DEBUG)

    subcommands = {
        'compile_cnn': compile_cnn._main,
        'consensuses': consensuses._main,
        'generate_cnn_outputs': generate_cnn_outputs._main,
        'generate_dataset': generate_dataset._main,
    }

    subcommands[args.subcommand](args)


if __name__ == '__main__':
    main()
