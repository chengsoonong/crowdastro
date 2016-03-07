"""From https://github.com/willettk/rgz-analysis, MIT licensed:

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

# How can I check for collinearity?

import numpy as np

def make_xy(n=10):

    x = np.arange(n)
    y = np.array([int(i) for i in np.random.random(n)*10])

    return x,y

def line(p1,p2):

    slope = (p1[1] - p2[1]) / (p1[0] - p2[0])
    offset = p1[1] - slope * p1[0]

    return slope,offset

def collinear(x,y,tol = 1e-6):

    all_collinear_points = set()

    #x,y = make_xy()
    #x= [  48.93867925,   43.04245283,   50.11792453]
    #y= [ 346.69811321,  340.80188679,  347.87735849]
    all_points = [(xx,yy) for xx,yy in zip(x,y)]

    sp = set(all_points)

    for p1 in sp:
        sp_not1 = sp.copy()
        sp_not1.remove(p1)
        for p2 in sp_not1:
            slope,offset = line(p1,p2)
            S = set()
            for p in sp_not1:
                snew,onew = line(p,p1)
                if abs(slope-snew) < tol and abs(offset - onew) < tol:
                    S.add(p)
            if len(S) >= 2:
                all_collinear_points.add(frozenset(S))
    
    return all_collinear_points

'''
from scipy.ndimage.filters import maximum_filter
from scipy import stats
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.linalg.basic import LinAlgError

def kernel_test():

    x_exists = [0,1,2.0001]
    y_exists = [0,1,2]

    x_exists = [  48.93867925,   43.04245283,   50.11792453]
    y_exists = [ 346.69811321,  340.80188679,  347.87735849]
    
    # X,Y = grid of uniform coordinates over the IR pixel plane
    X, Y = np.mgrid[1:50,1:50]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([x_exists, y_exists])
    try:
        kernel = stats.gaussian_kde(values)
    except LinAlgError:
        print 'LinAlgError in KD estimation'
    
    return kernel(positions)
'''


