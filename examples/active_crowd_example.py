import logging
import os.path
import sys

import matplotlib.pyplot as plt
import numpy
import sklearn.cluster
import sklearn.datasets

sys.path.insert(1, os.path.join('..'))
import crowdastro.crowd.yan_sparse as yan_sparse

logging.captureWarnings(True)

# Generate some data.
n_annotators, n_dim, n_samples = 4, 2, 50
x, z = sklearn.datasets.make_classification(
        n_samples=n_samples,
        n_features=n_dim,
        n_informative=n_dim,
        n_redundant=0,
        flip_y=0.1,
        class_sep=2.0,
        random_state=10)
z = z.reshape((-1, 1)).astype(bool)

# Generate annotator labels. Cluster the data using k-means, then let each
# annotator be an expert on one cluster. Switch labels at random 35% of the time
# for all other clusters.
y = numpy.ma.MaskedArray(numpy.zeros((n_annotators, n_samples), dtype=bool))
k_means = sklearn.cluster.KMeans(n_clusters=n_annotators) 
clusters = k_means.fit_predict(x)
for annotator in range(n_annotators):
    for sample in range(n_samples):
        if clusters[sample] == annotator or numpy.random.random() > 0.35:
            y[annotator, sample] = z[sample]
        else:
            # Flip labels at random for non-expert regions.
            y[annotator, sample] = not z[sample]

# Randomly mask 50% of the labels.
y.mask = numpy.random.binomial(1, 0.8, size=y.shape)

# Train and predict using the passive crowd algorithm.
a, b, *h = yan_sparse.train(x, y, lr_init=True, trials=15)
predictions = yan_sparse.predict(a, b, x)

## Plots ##

fig = plt.figure()

# Plot the groundtruth.
ax = fig.add_subplot(2, 3, 1)
ax.set_title('Groundtruth')
ax.scatter(x[z.ravel() == 0, 0], x[z.ravel() == 0, 1], c='blue', marker='x',
           s=20)
ax.scatter(x[z.ravel() == 1, 0], x[z.ravel() == 1, 1], c='red', marker='x',
           s=20)

# Plot each annotator.
# zz is used to plot the regions of expertise.
xx, yy = numpy.meshgrid(numpy.linspace(x[:, 0].min(), x[:, 0].max()),
                        numpy.linspace(x[:, 1].min(), x[:, 1].max()))
zz = k_means.predict(numpy.vstack([xx.ravel(), yy.ravel()]).T)
zz = zz.reshape(xx.shape)
for t in range(4):
    ax = fig.add_subplot(2, 3, 2 + t)
    ax.set_title('Annotator {}'.format(t + 1))

    # Plot the region of expertise.
    ax.contourf(xx, yy, zz == t, cmap='Paired', alpha=0.4)
    obs = (~y[t].mask).nonzero()[0]
    ax.scatter(x[numpy.logical_and(y[t] == 0, y[t].mask == 0), 0],
               x[numpy.logical_and(y[t] == 0, y[t].mask == 0), 1],
               c='blue', marker='x', s=20)
    ax.scatter(x[numpy.logical_and(y[t] == 1, y[t].mask == 0), 0],
               x[numpy.logical_and(y[t] == 1, y[t].mask == 0), 1],
               c='red', marker='x', s=20)

# Plot predictions.
ax = fig.add_subplot(2, 3, 6)
ax.set_title('Predictions')
ax.scatter(x[predictions.ravel() == 0, 0], x[predictions.ravel() == 0, 1],
           c='blue', marker='x', s=20)
ax.scatter(x[predictions.ravel() == 1, 0], x[predictions.ravel() == 1, 1],
           c='red', marker='x', s=20)

plt.show()
