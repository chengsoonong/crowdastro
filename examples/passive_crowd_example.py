import logging
import os.path
import sys

import matplotlib.pyplot as plt
import numpy
import sklearn.cluster
import sklearn.datasets

sys.path.insert(1, os.path.join('..', 'crowdastro', 'active_learning'))
import passive_crowd

logging.captureWarnings(True)

# Generate some data.
n_annotators, n_dim, n_samples = 100, 20, 2000
x, z = sklearn.datasets.make_classification(
        n_samples=n_samples,
        n_features=n_dim,
        n_informative=n_dim,
        n_redundant=0,
        flip_y=0.1,
        class_sep=1.0,
        random_state=10)
z = z.reshape((-1, 1)).astype(bool)

def run_example(x, z, n_annotators, n_dim, n_samples, plot=False):
    # Generate annotator labels. Cluster the data using k-means, then let each
    # annotator be an expert on one cluster. Switch labels at random 35% of the time
    # for all other clusters.
    y = numpy.zeros((n_annotators, n_samples), dtype=bool)
    k = 5
    k_means = sklearn.cluster.KMeans(n_clusters=k) 
    clusters = k_means.fit_predict(x)
    for annotator in range(n_annotators):
        for sample in range(n_samples):
            if clusters[sample] == annotator % k or numpy.random.random() > 0.5:
                y[annotator, sample] = z[sample]
            else:
                # Flip labels at random for non-expert regions.
                y[annotator, sample] = not z[sample]

    # Train and predict using the passive crowd algorithm.
    a, b, w, g = passive_crowd.train(x, y, skip_zeros=True, lr_init=True)
    predictions = passive_crowd.predict(a, b, x)

    if plot:
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
        # zz = k_means.predict(numpy.vstack([xx.ravel(), yy.ravel()]).T)
        # zz = zz.reshape(xx.shape)
        for t in range(4):
            ax = fig.add_subplot(2, 3, 2 + t)
            ax.set_title('Annotator {}'.format(t + 1))

            # Plot the region of expertise.
            # ax.contourf(xx, yy, zz == t, cmap='Paired', alpha=0.4)
            ax.scatter(x[y[t] == 0, 0], x[y[t] == 0, 1], c='blue', marker='x', s=20)
            ax.scatter(x[y[t] == 1, 0], x[y[t] == 1, 1], c='red', marker='x', s=20)

        # Plot predictions.
        ax = fig.add_subplot(2, 3, 6)
        ax.set_title('Predictions')
        ax.scatter(x[predictions.ravel() == 0, 0], x[predictions.ravel() == 0, 1],
                   c='blue', marker='x', s=20)
        ax.scatter(x[predictions.ravel() == 1, 0], x[predictions.ravel() == 1, 1],
                   c='red', marker='x', s=20)

        plt.show()

    eq = predictions.ravel() == z.ravel()
    err = eq.mean()
    return err


def run_many(n, x, z, n_annotators, n_dim, n_samples):
    errs = []
    for _ in range(n):
        errs.append(run_example(x, z, n_annotators, n_dim, n_samples))

    return numpy.mean(errs), numpy.std(errs)


if __name__ == '__main__':
    mean, stdev = run_many(10, x, z, n_annotators, n_dim, n_samples)
    print(mean, stdev)