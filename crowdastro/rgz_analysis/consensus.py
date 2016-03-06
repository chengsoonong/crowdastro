#!/usr/bin/env python3
"""Computes consensus for RGZ classifications.

Heavily based on https://github.com/willettk/rgz-analysis, MIT licensed:

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
"""

import argparse
import collections
import csv
import datetime
import functools
import io
import json
import logging
import operator
import os
import os.path
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy
import pandas
import pymongo
import scipy.ndimage.filters
import scipy.stats
import scipy.ndimage.morphology
import scipy.linalg.basic

import collinearity
import load_contours

# Mongo DB server.
HOST = 'localhost'
PORT = 27017
DB_NAME = 'radio'

# Filename stem of output files.
PATH_STEM = "consensus_rgz_atlas"

# Regex to match whitespace.
WHITESPACE_REGEX = re.compile(r'\s+')

# Main release date of RGZ data.
MAIN_RELEASE_DATE = datetime.datetime(2013, 12, 17, 0, 0, 0, 0)

# Number of pixels in original JPG image.
IMG_HEIGHT_OLD = 424.0
IMG_WIDTH_OLD = 424.0

# Number of pixels in downloaded JPG image.
IMG_HEIGHT_NEW = 500.0
IMG_WIDTH_NEW = 500.0

# Number of pixels in the FITS image (?).
# TODO(MatthewJA): What are these values? Can we pull them from the FITS
# headers?
FITS_HEIGHT = 301.0
FITS_WIDTH = 301.0
FIRST_FITS_HEIGHT = 132.0
FIRST_FITS_WIDTH = 132.0

# Arcseconds/pixel in the FITS image.
PIXEL_SIZE = 0.00016667

# TODO(MatthewJA): Add parameters for ATLAS - rgz-analysis apparently only uses
# FIRST.

# Setup MongoDB.
client = pymongo.MongoClient(HOST, PORT)
db = client[DB_NAME]
subjects = db.radio_subjects  # RGZ examples.
classifications = db.radio_classifications  # classifications of each subject.
subindex = classifications.create_index([('subject_ids',pymongo.ASCENDING)],
                                        name='subject_ids_1')

# Bounding pixel indices.
XMIN = 1.
XMAX = IMG_HEIGHT_NEW
YMIN = 1.
YMAX = IMG_WIDTH_NEW

BAD_KEYS = ('finished_at', 'started_at', 'user_agent', 'lang', 'pending')

rgz_dir = None  # Set after argument parsing.

def checksum(zid,experts_only=False,excluded=[],no_anonymous=False,write_peak_data=True):
    """Find the consensus for all users who have classified a particular galaxy.

    This function is taken (almost) verbatim from willettk/rgz-analysis. Minor
    changes: Updated to Python 3, and swapped print for logging.
    """

    sub = subjects.find_one({'zooniverse_id':zid})
    imgid = sub['_id']

    # Classifications for this subject after launch date
    class_params = {"subject_ids": imgid, "updated_at": {"$gt": MAIN_RELEASE_DATE}}
    # Only get the consensus classification for the science team members
    if experts_only:
        class_params['expert'] = True

    # If comparing a particular volunteer (such as an expert), don't include self-comparison
    if len(excluded) > 0:
        class_params['user_name'] = {"$nin":excluded}

    '''
    # To exclude the experts:
    class_params['expert'] = {"$exists":False}
    '''

    # To exclude anonymous classifications (registered users only):
    if no_anonymous:
        if 'user_name' in class_params:
            class_params['user_name']["$exists"] = True
        else:
            class_params['user_name'] = {"$exists":True}

    _c = classifications.find(class_params)

    # Empty dicts and lists 
    cdict = {}

    unique_users = set()
    
    clen_start = 0
    clist_all = []
    listcount = []

    # Compute the most popular combination for each NUMBER of galaxies identified in image
    
    for c in _c:

        clist_all.append(c)
        clen_start += 1
        
        # Skip classification if they already did one?

        try:
            user_name = c['user_name']
        except KeyError:
            user_name = 'Anonymous'

        if user_name not in unique_users or user_name is 'Anonymous':

            unique_users.add(user_name)
            listcount.append(True)
        
            sumlist = []    # List of the checksums over all possible combinations

            # Only find data that was an actual marking, not metadata
            goodann = [x for x in c['annotations'] if (list(x.keys())[0] not in BAD_KEYS)]
            n_galaxies = len(goodann)
    
            if n_galaxies > 0:  # There must be at least one galaxy!
                for idx,ann in enumerate(goodann):
    
                    xmaxlist = []
                    try:
                        radio_comps = ann['radio']

                        # loop over all the radio components within an galaxy
                        if radio_comps != 'No Contours':
                            for rc in radio_comps:
                                xmaxlist.append(float(radio_comps[rc]['xmax']))
                        # or make the value -99 if there are no contours
                        else:
                            xmaxlist.append(-99)
                    except KeyError:
                        xmaxlist.append(-99)
    
                    # To create a unique ID for the combination of radio components,
                    # take the product of all the xmax coordinates and sum them together.
                    product = functools.reduce(operator.mul, xmaxlist, 1)
                    sumlist.append(round(product,3))

                checksum = sum(sumlist)
            else:
                checksum = -99

            c['checksum'] = checksum
    
            # Insert checksum into dictionary with number of galaxies as the index
            if n_galaxies in cdict:
                cdict[n_galaxies].append(checksum)
            else:
                cdict[n_galaxies] = [checksum]

        else:
            listcount.append(False)
            #print 'Removing classification for %s' % user_name
    
    # Remove duplicates and classifications for no object
    clist = [c for lc,c in zip(listcount,clist_all) if lc and c['checksum'] != -99]

    clen_diff = clen_start - len(clist)

    '''
    if clen_diff > 0:
        print '\nSkipping %i duplicated classifications for %s. %i good classifications total.' % (clen_diff,zid,len(clist))
    '''

    maxval=0
    mc_checksum = 0.

    # Find the number of galaxies that has the highest number of consensus classifications

    for k,v in cdict.items():
        mc = collections.Counter(v).most_common()
        # Check if the most common selection coordinate was for no radio contours
        if mc[0][0] == -99.0:
            if len(mc) > 1:
                # If so, take the selection with the next-highest number of counts
                mc_best = mc[1]
            else:
                continue
        # Selection with the highest number of counts
        else:
            mc_best = mc[0]
        # If the new selection has more counts than the previous one, choose it as the best match;
        # if tied or less than this, remain with the current consensus number of galaxies
        if mc_best[1] > maxval:
            maxval = mc_best[1]
            mc_checksum = mc_best[0]
    
    # Find a galaxy that matches the checksum (easier to keep track as a list)

    try:
        cmatch = next(i for i in clist if i['checksum'] == mc_checksum)
    except StopIteration:
        # Necessary for objects like ARG0003par; one classifier recorded 22 "No IR","No Contours" in a short space. Still shouldn't happen.
        logging.warning('No non-zero classifications recorded for %s' % zid)
        return
   
    # Find IR peak for the checksummed galaxies
    
    goodann = [x for x in cmatch['annotations'] if list(x.keys())[0] not in BAD_KEYS]

    # Find the sum of the xmax coordinates for each galaxy. This gives the index to search on.
    
    cons = {}
    cons['zid'] = zid
    cons['source'] = sub['metadata']['source']
    ir_x,ir_y = {},{}
    cons['answer'] = {}
    cons['n_users'] = maxval
    cons['n_total'] = len(clist)

    answer = cons['answer']

    for k,gal in enumerate(goodann):
        xmax_temp = []
        bbox_temp = []
        try:
            for v in gal['radio'].values():
                xmax_temp.append(float(v['xmax']))
                bbox_temp.append((v['xmax'],v['ymax'],v['xmin'],v['ymin']))
            checksum2 = round(sum(xmax_temp),3)
            answer[checksum2] = {}
            answer[checksum2]['ind'] = k
            answer[checksum2]['xmax'] = xmax_temp
            answer[checksum2]['bbox'] = bbox_temp
        except KeyError:
            logging.warning('KeyError for %s, %s.', gal, zid)
        except AttributeError:
            logging.warning('No Sources, No IR recorded for %s.', zid)
    
        # Make empty copy of next dict in same loop
        ir_x[k] = []
        ir_y[k] = []
    
    # Now loop over all sets of classifications to get the IR counterparts
    for c in clist:
        if c['checksum'] == mc_checksum:
    
            annlist = [ann for ann in c['annotations'] if list(ann.keys())[0] not in BAD_KEYS]
            for ann in annlist:
                if 'ir' in list(ann.keys()):
                    # Find the index k that this corresponds to
                    try:
                        xmax_checksum = round(sum([float(ann['radio'][a]['xmax']) for a in ann['radio']]),3)
                    except TypeError:
                        xmax_checksum = -99

                    try:
                        k = answer[xmax_checksum]['ind']

                        if ann['ir'] == 'No Sources':
                            ir_x[k].append(-99)
                            ir_y[k].append(-99)
                        else:
                            # Only takes the first IR source right now; NEEDS TO BE MODIFIED.

                            ir_x[k].append(float(ann['ir']['0']['x']))
                            ir_y[k].append(float(ann['ir']['0']['y']))
                    except KeyError:
                        logging.warning('"No radio" still appearing as valid consensus option.')

    # Perform a kernel density estimate on the data for each galaxy
    
    scale_ir = IMG_HEIGHT_NEW/IMG_HEIGHT_OLD

    peak_data = []

    # Remove empty IR peaks if they exist

    for (xk,xv),(yk,yv) in zip(list(ir_x.items()), list(ir_y.items())):
        
        if len(xv) == 0:
            ir_x.pop(xk)
        if len(yv) == 0:
            ir_y.pop(yk)

    assert len(ir_x) == len(ir_y),'Lengths of ir_x (%i) and ir_y (%i) are not the same' % (len(ir_x),len(ir_y))

    for (xk,xv),(yk,yv) in zip(list(ir_x.items()), list(ir_y.items())):
        
        if len(xv) == 0:
            irx

        pd = {}
    
        x_exists = [xt * scale_ir for xt in xv if xt != -99.0]
        y_exists = [yt * scale_ir for yt in yv if yt != -99.0]

        x_all = [xt * scale_ir for xt in xv]
        y_all = [yt * scale_ir for yt in yv]
        coords_all = [(xx,yy) for xx,yy in zip(x_all,y_all)]
        ir_counter = collections.Counter(coords_all)
        most_common_ir = ir_counter.most_common(1)[0][0]

        if len(collections.Counter(x_exists)) > 2 and len(collections.Counter(y_exists)) > 2 and most_common_ir != (-99,-99):

            # X,Y = grid of uniform coordinates over the IR pixel plane
            X, Y = numpy.mgrid[XMIN:XMAX, YMIN:YMAX]
            positions = numpy.vstack([X.ravel(), Y.ravel()])
            try:
                values = numpy.vstack([x_exists, y_exists])
            except ValueError:
                # Breaks on the tutorial subject. Find out why len(x) != len(y)
                logging.warning('Length of IR x array: %i; Length of IR y array: %i (zid %s)', len(x_exists), len(y_exists), zid)
            try:
                kernel = scipy.stats.gaussian_kde(values)
            except scipy.linalg.basic.LinAlgError:
                logging.debug('LinAlgError in KD estimation for %s', zid)
                continue

            # Even if there are more than 2 sets of points, if they are mutually co-linear, 
            # matrix can't invert and kernel returns NaNs. 

            kp = kernel(positions)

            if numpy.isnan(kp).sum() > 0:
                acp = collinearity.collinear(x_exists,y_exists)
                if len(acp) > 0:
                    logging.warning('There are %i unique points for %s (source no. %i in the field), but all are co-linear; KDE estimate does not work.' % (len(collections.Counter(x_exists)),zid,xk))
                else:
                    logging.warning('There are NaNs in the KDE for %s (source no. %i in the field), but points are not co-linear.' % (zid,xk))

                for k,v in answer.items():
                    if v['ind'] == xk:
                        answer[k]['ir'] = (numpy.mean(x_exists),numpy.mean(y_exists))
        
            else:

                Z = numpy.reshape(kp.T, X.shape)
                
                # Find the number of peaks
                # http://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array
                
                neighborhood = numpy.ones((10,10))
                local_max = scipy.ndimage.filters.maximum_filter(Z, footprint=neighborhood)==Z
                background = (Z==0)
                eroded_background = scipy.ndimage.morphology.binary_erosion(background, structure=neighborhood, border_value=1)
                detected_peaks = local_max ^ eroded_background
                
                npeaks = detected_peaks.sum()
    
                #return X,Y,Z,npeaks
    
                pd['X'] = X
                pd['Y'] = Y
                pd['Z'] = Z
                pd['npeaks'] = npeaks

                try:
                    xpeak = float(pd['X'][pd['Z']==pd['Z'].max()][0])
                    ypeak = float(pd['Y'][pd['Z']==pd['Z'].max()][0])
                except IndexError:
                    logging.warning('IndexError for %s, %s, %s.', pd, zid, clist)

                for k,v in answer.items():
                    if v['ind'] == xk:
                        answer[k]['ir_peak'] = (xpeak,ypeak)
                        # Don't write to consensus for serializable JSON object 
                        if write_peak_data:
                            answer[k]['peak_data'] = pd
                            answer[k]['ir_x'] = x_exists
                            answer[k]['ir_y'] = y_exists
        else:

            # Note: need to actually put a limit in if less than half of users selected IR counterpart.
            # Right now it still IDs a sources even if only 1/10 users said it was there.

            for k,v in answer.items():
                if v['ind'] == xk:
                    # Case 1: multiple users selected IR source, but not enough unique points to pinpoint peak
                    if most_common_ir != (-99,-99) and len(x_exists) > 0 and len(y_exists) > 0:
                        answer[k]['ir'] = (x_exists[0],y_exists[0])
                    # Case 2: most users have selected No Sources
                    else:
                        answer[k]['ir'] = (-99,-99)

    return cons

def run_sample(data_path, catalogue_path, limit=0):
    """Runs the consensus algorithm described in Banfield et al., 2015."""
    # TODO(MatthewJA): Reimplement the 'update' argument.
    # TODO(MatthewJA): Reimplement the 'do_plot' argument.
    # TODO(MatthewJA): This only works for ATLAS subjects.
    # TODO(MatthewJA): The original code only worked on FIRST subjects. Have I
    # broken anything by running ATLAS subjects through it?
    paths = load_contours.make_pathdict(data_path, catalogue_path)

    sample_subjects = [cz for cz in subjects.find({
        'metadata.survey': 'atlas'
    }).limit(limit)]
    logging.debug('Found {} subjects.'.format(len(sample_subjects)))

    zooniverse_ids = [cz['zooniverse_id'] for cz in sample_subjects]

    with open('%s/csv/%s.csv' % (rgz_dir, PATH_STEM), 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            'zooniverse_id',
            'first_id',
            'n_users',
            'n_total',
            'consensus_level',
            'n_radio',
            'label',
            'bbox',
            'ir_peak',
        ])

        for idx, zid in enumerate(zooniverse_ids):
            logging.debug('Zooniverse ID: {}'.format(zid))

            if not idx % 100:
                # Progress.
                now = datetime.datetime.now().strftime('%H:%M:%S.%f')
                progress = idx / len(zooniverse_ids)
                logging.info('{:.02%} {}'.format(progress, now))

            cons = checksum(zid)
            cons['consensus_level'] = cons['n_users'] / cons['n_total']

            # CSV.
            for ans in cons['answer'].values():
                try:
                    ir_peak = ans['ir_peak']
                except KeyError:
                    ir_peak = ans.get('ir', None)

                writer.writerow([
                    cons['zid'],
                    cons['source'],
                    cons['n_users'],
                    cons['n_total'],
                    cons['consensus_level'],
                    len(ans['xmax']),
                    alpha(ans['ind']),
                    bbox_unravel(ans['bbox']),
                    ir_peak,
                ])

    # CSV
    cmaster = pandas.read_csv('%s/csv/%s.csv' % (rgz_dir, PATH_STEM))
    cmaster75 = cmaster[cmaster['consensus_level'] >= 0.75]
    cmaster75.to_csv('%s/csv/%s_75.csv' % (rgz_dir, PATH_STEM), index=False)

    logging.info('Completed consensus.')

def bbox_unravel(bbox):
    """Turns a list of (str, str) tuples into (float, float) tuples.

    bbox: Array of (str, str) tuples.
    """

    bboxes = []
    for lobe in bbox:
        t = [float(x) for x in lobe]
        t = tuple(t)
        bboxes.append(t)

    return bboxes

def alpha(i):
    return chr(i % 26 + ord('a')) * (i // 26 + 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('output', help='output directory')
    parser.add_argument('data', help='location of CDFS and ELAIS data')
    parser.add_argument('catalogue', help='path to ATLAS component catalogue')
    parser.add_argument('--update', action='store_true',
                        help='update instead of rerunning all')

    args = parser.parse_args()

    # try:
    try:
        os.makedirs(os.path.join(args.output, 'csv'))
    except OSError:
        # Already exists.
        pass

    try:
        os.mkdir(os.path.join(args.output, 'json'))
    except OSError:
        # Already exists.
        pass

    rgz_dir = args.output

    logging.basicConfig(level=logging.INFO)

    logging.info('Starting at %s',
                 datetime.datetime.now().strftime('%H:%M:%S.%f'))
    run_sample(args.data, args.catalogue)
    logging.info('Finished at %s',
                 datetime.datetime.now().strftime('%H:%M:%S.%f'))
