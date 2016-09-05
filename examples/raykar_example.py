import logging
import os.path
import sys

import matplotlib.pyplot as plt
import numpy
import sklearn.cluster
import sklearn.datasets

sys.path.insert(1, os.path.join('..', 'crowdastro', 'active_learning'))
import raykar

logging.captureWarnings(True)

# Generate some data.
n_labellers, n_dim, n_examples = 4, 2, 50
x, z = sklearn.datasets.make_classification(
        n_samples=n_examples,
        n_features=n_dim,
        n_informative=n_dim,
        n_redundant=0,
        flip_y=0.0,
        class_sep=1.0,
        random_state=50)
z = z.reshape((-1, 1)).astype(bool)

def run_example(x, z, n_labellers, n_dim, n_examples, plot=False):
    # Generate annotator labels. Cluster the data using k-means, then let each
    # annotator be an expert on one cluster. Switch labels at random 35% of the
    # time for all other clusters.
    y = numpy.zeros((n_labellers, n_examples), dtype=bool)
    k = 5
    k_means = sklearn.cluster.KMeans(n_clusters=k) 
    clusters = k_means.fit_predict(x)
    accuracies = [0.1, 0.4, 0.8, 1.0]
    for annotator in range(n_labellers):
        for sample in range(n_examples):
            if (clusters[sample] == annotator % k or
                    numpy.random.random() < accuracies[annotator]):
                y[annotator, sample] = z[sample]
            else:
                # Flip labels at random for non-expert regions.
                y[annotator, sample] = not z[sample]

    # Randomly hide half the labels.
    y = numpy.ma.MaskedArray(y,
            mask=False)#numpy.random.binomial(1, 0.5, size=y.shape))

    # Train and predict using the passive crowd algorithm.
    a, b, w = raykar.train(x, y)
    predictions = raykar.predict(w, x)

    print('a: {}'.format(a))
    print('b: {}'.format(b))

    if plot:
        ## Plots ##

        fig = plt.figure()

        # Plot the groundtruth.
        ax = fig.add_subplot(2, 3, 1)
        ax.set_title('Groundtruth')
        ax.scatter(x[z.ravel() == 0, 0], x[z.ravel() == 0, 1], c='blue',
                   marker='x', s=20)
        ax.scatter(x[z.ravel() == 1, 0], x[z.ravel() == 1, 1], c='red',
                   marker='x', s=20)

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
            ax.scatter(x[y[t] == 0, 0], x[y[t] == 0, 1], c='blue', marker='x',
                       s=20)
            ax.scatter(x[y[t] == 1, 0], x[y[t] == 1, 1], c='red', marker='x',
                       s=20)

        # Plot predictions.
        ax = fig.add_subplot(2, 3, 6)
        ax.set_title('Predictions')
        ax.scatter(x[predictions.ravel() == 0, 0],
                   x[predictions.ravel() == 0, 1],
                   c='blue', marker='x', s=20)
        ax.scatter(x[predictions.ravel() == 1, 0],
                   x[predictions.ravel() == 1, 1],
                   c='red', marker='x', s=20)

        plt.show()

    eq = predictions.ravel() == z.ravel()
    err = eq.mean()
    return err


if __name__ == '__main__':
    logging.root.setLevel(logging.DEBUG)
    run_example(x, z, n_labellers, n_dim, n_examples, plot=True)
