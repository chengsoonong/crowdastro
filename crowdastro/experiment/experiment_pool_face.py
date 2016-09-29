"""Pools a face image.

Matthew Alger
The Australian National University
2016
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy
import scipy.misc
import skimage.measure

face = scipy.misc.face(gray=True)
max_pooled = skimage.measure.block_reduce(face, (20, 20), numpy.max)
min_pooled = skimage.measure.block_reduce(face, (20, 20), numpy.min)
mean_pooled = skimage.measure.block_reduce(face, (20, 20), numpy.mean)

fig = plt.figure()

# Original
ax = fig.add_subplot(2, 2, 1)
ax.imshow(face, cmap='gray')
ax.axis('off')

# Max pooled
ax = fig.add_subplot(2, 2, 2)
ax.imshow(max_pooled[:-1, :-1], cmap='gray')
ax.axis('off')

# Min pooled
ax = fig.add_subplot(2, 2, 3)
ax.imshow(min_pooled[:-1, :-1], cmap='gray')
ax.axis('off')

# Mean pooled
ax = fig.add_subplot(2, 2, 4)
ax.imshow(mean_pooled[:-1, :-1], cmap='gray')
ax.axis('off')

plt.show()

for im in [face, max_pooled, min_pooled, mean_pooled]:
    plt.imshow(im[:-1, :-1], cmap='gray', interpolation='None')
    plt.axis('off')
    plt.show()
