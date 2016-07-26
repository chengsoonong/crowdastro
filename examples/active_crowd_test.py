import sys
import logging

import matplotlib.pyplot as plt
import numpy
import sklearn.datasets

sys.path.insert(1, '..')
import crowdastro.active_learning.active_crowd as active_crowd

n_annotators, n_dim, n_samples = 4, 2, 50
x, z = sklearn.datasets.make_classification(
        n_samples=n_samples,
        n_features=n_dim,
        n_informative=n_dim,
        n_redundant=0,
        random_state=83)
z = z.reshape((-1, 1))
y = numpy.ma.MaskedArray(z.repeat(n_annotators, axis=1).T)
if True:
    # Annotator 1 is really bad at this.
    y[0, :] = numpy.ones((n_samples,))
    # Annotator 2 is wrong 10% of the time.
    indices = numpy.arange(n_samples)
    numpy.random.shuffle(indices)
    indices = indices[:n_samples // 10]
    indices.sort()
    y[1, indices] = 1 - y[1, indices]
    # Annotator 3 is good, but only if x[0] > 0.
    y[2, x[:, 0] <= 0] = numpy.round(
            numpy.random.uniform(size=((x[:, 0] <= 0).sum(),)))
    # Annotator 4 is wrong 30% of the time.
    indices = numpy.arange(n_samples)
    numpy.random.shuffle(indices)
    indices = indices[:n_samples * 3 // 10]
    indices.sort()
    y[3, indices] = 1 - y[3, indices]
# Hide 80% of labels at random to simulate partial annotation.
# Each labeller will be labelling about 15 examples.
y.mask = numpy.random.binomial(1, 0.7, size=y.shape)
assert y.shape == (n_annotators, n_samples)

logging.root.setLevel(logging.DEBUG)
logging.captureWarnings(True)
a, b, w, g = active_crowd.train(x, y)
print('w:', w)
print('g:', g)
predictions = active_crowd.predict(a, b, x)
plt.subplot(2, 3, 1)
plt.title('Groundtruth')
plt.scatter(x[z.ravel() == 0, 0], x[z.ravel() == 0, 1], c='blue', marker='x')
plt.scatter(x[z.ravel() == 1, 0], x[z.ravel() == 1, 1], c='red', marker='x')
for t in range(n_annotators):
    if t >= 2:
        plt.subplot(2, 3, 2 + t + 1)
    else:
        plt.subplot(2, 3, 2 + t)
    plt.title('Annotator {}'.format(t + 1))
    plt.scatter(x[y[t][~y[t].mask] == 0, 0], x[y[t][~y[t].mask] == 0, 1], c='blue', marker='x')
    plt.scatter(x[y[t][~y[t].mask] == 1, 0], x[y[t][~y[t].mask] == 1, 1], c='red', marker='x')
plt.subplot(2, 3, 4)
plt.title('Predictions')
plt.scatter(x[predictions.ravel() == 0, 0], x[predictions.ravel() == 0, 1], c='blue', marker='x')
plt.scatter(x[predictions.ravel() == 1, 0], x[predictions.ravel() == 1, 1], c='red', marker='x')
plt.show()