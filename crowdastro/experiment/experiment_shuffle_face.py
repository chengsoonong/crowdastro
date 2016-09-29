"""Shuffles a face image.

Matthew Alger
The Australian National University
2016
"""

import matplotlib.pyplot as plt
import numpy
import scipy.misc

face = scipy.misc.face(gray=True)
shuffled_face = face.copy()
print(shuffled_face.shape)
numpy.random.shuffle(shuffled_face)
shuffled_face = shuffled_face.T
numpy.random.shuffle(shuffled_face)
shuffled_face = shuffled_face.T

fig = plt.figure()

# Original
ax = fig.add_axes([0.05, 0.20, 0.40, 0.60], frame_on=False)
ax.imshow(face, cmap='gray')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

# Shuffled
ax = fig.add_axes([0.55, 0.20, 0.40, 0.60], frame_on=False)
ax.imshow(shuffled_face, cmap='gray')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

plt.show()

# Some individual plots.
plt.imshow(face, cmap='gray')
plt.axis('off')
plt.show()
plt.imshow(shuffled_face, cmap='gray')
plt.axis('off')
plt.show()
