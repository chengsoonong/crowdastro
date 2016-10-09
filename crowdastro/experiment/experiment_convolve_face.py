"""Convolves a face image.

Matthew Alger
The Australian National University
2016
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy
import scipy.misc
import scipy.signal
import skimage.exposure

face = scipy.misc.face(gray=True)
filter_ = numpy.array([[-1, 1, 0],
                       [-2, 2, 0],
                       [-1, 1, 0]])
filtered = scipy.signal.convolve2d(
    face, filter_, mode='same', boundary='fill', fillvalue=0)

fig = plt.figure()

# Original
ax = fig.add_axes([0.05, 0.30, 0.40, 0.60], frame_on=False)
ax.imshow(face, cmap='gray')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

# Filtered
ax = fig.add_axes([0.55, 0.30, 0.40, 0.60], frame_on=False)
p2, p98 = numpy.percentile(filtered, (2, 98))
filtered = skimage.exposure.rescale_intensity(filtered, in_range=(p2, p98))
ax.imshow(filtered, cmap='gray')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

# Filter
ax = fig.add_axes([0.35, 0.05, 0.30, 0.30], frame_on=False)
ax.imshow(filter_, cmap='gray', interpolation='None')
for axis in [ax.xaxis, ax.yaxis]:
    axis.set_major_locator(ticker.MaxNLocator(integer=True))

plt.show()

# Some individual plots.
plt.imshow(face, cmap='gray')
plt.axis('off')
plt.show()
plt.imshow(filtered, cmap='gray')
plt.axis('off')
plt.show()
plt.imshow(filter_, cmap='gray', interpolation='None')
plt.axis('off')
plt.show()
