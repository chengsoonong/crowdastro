"""Command-line interface for crowdastro.

Matthew Alger
The Australian National University
2016
"""

import argparse
import sys

from . import __description__
from . import compile_cnn

def main():
    parser = argparse.ArgumentParser(description=__description__)
    subparsers = parser.add_subparsers(dest='subcommand')

    parser_compile_cnn = subparsers.add_parser('compile_cnn',
            help='compile a convolutional neural network')
    compile_cnn._populate_parser(parser_compile_cnn)

    # http://stackoverflow.com/a/11287731/1105803
    if len(sys.argv) < 2:
        sys.argv.append('--help')

    args = parser.parse_args()

    subcommands = {
        'compile_cnn': compile_cnn._main
    }

    subcommands[args.subcommand](args)


if __name__ == '__main__':
    main()
