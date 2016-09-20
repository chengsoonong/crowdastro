"""Compute checksums for a list of files.

Matthew Alger
The Australian National University
2016
"""

import argparse
import csv
import io

from crowdastro.import_data import hash_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='+')

    args = parser.parse_args()

    out = io.StringIO()
    writer = csv.writer(out)

    for filename in args.filenames:
        with open(filename, 'rb') as f:
            h = hash_file(f)
            writer.writerow((filename, h))

    print(out.getvalue())


if __name__ == '__main__':
    main()
