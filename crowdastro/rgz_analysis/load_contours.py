'''Find paths for RGZ data and use it to load raw data such as contour files

Looks in local directories first; if not found, sends request over network to AWS

Kyle Willett, UMN

From https://github.com/willettk/rgz-analysis, MIT licensed:

The MIT License (MIT)

Copyright (c) 2014 Kyle Willett

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import os
import json
import re
import requests

WHITESPACE_REGEX = re.compile(r'\s+')

def make_pathdict(data_path, catalogue_path):
    """Generates a dictionary mapping subject.metadata.source to paths.

    data_path: Path to the CDFS and ELAIS data. This directory should contain
        two folders, cdfs and elais, which then each contain two folders 2x2
        and 5x5. These folders will then contain the FITS data in the form
        {ID}_{ir, radio}.fits. The data directory should also contain rgz_cache,
        which will cache files downloaded from the RGZ server.
    catalogue_path: Path to the ATLAS DR3 component catalogue.
    """
    pathdict = {}
    surveys = {
        'ELAIS-S1': 'elais',
        'CDFS': 'cdfs',
    }

    with open(catalogue_path) as catalogue_file:
        for line in catalogue_file:
            if line.startswith('#'):
                # Skip comments.
                continue

            line = WHITESPACE_REGEX.split(line.strip())

            # The first column is the ID, and the second is the name.
            # subject.metadata.source refers to the name, but the filepaths
            # refer to the ID.
            sid = line[0]
            name = line[1]

            # The last column is the field identifier, which should be
            # either ELAIS-S1 or CDFS.
            field = line[24]
            if field not in surveys:
                raise ValueError(
                    'Unexpected field in ATLAS catalogue: {}'.format(field))

            base_path = os.path.join(data_path, surveys[field], '2x2')
            contours_path = os.path.join(data_path, 'rgz_cache',
                                         '{}_contours.json')
            ir_path = os.path.join(base_path, '{}_ir.fits')
            radio_path = os.path.join(base_path, '{}_radio.fits')

            pathdict[name] = {
                'contours': contours_path,
                'ir': ir_path,
                'radio': radio_path,
            }

    return pathdict

def get_contours(subject,pathdict):

    # Return the radio contours for an RGZ subject, input as a dictionary from MongoDB

    source = subject['metadata']['source']
    
    try:
        with open(pathdict[source]['contours']) as f:
            contours = json.load(f)
    # If pathdict is None, try to download over network
    except TypeError:
        r = requests.get(subject['location']['contours'])
        contours = r.json()
    
    return contours

