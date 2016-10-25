"""Plots a colour-colour diagram.

Matthew Alger
The Australian National University
2016
"""

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy
import sklearn.linear_model

VIRIDIS_PURPLE = [0.267004, 0.004874, 0.329415]


def main(training_path):
    with h5py.File(training_path, 'r') as training_h5:
        assert training_h5.attrs['ir_survey'] == 'wise'
        features = training_h5['features']
        labels = training_h5['labels'].value
        lr = sklearn.linear_model.LogisticRegression(class_weight='balanced')
        lr.fit(features, labels)
        probs = lr.predict_proba(features)[:, 1]
        w1_w2 = -2.5*numpy.log10(features[:, 4])
        w2_w3 = -2.5*numpy.log10(features[:, 5])

        pos = labels == 1  # probs > 0.5
        neg = labels == 0  # probs < 0.5

        probs = lr.predict_proba(features)[:, 1]

        matplotlib.rcParams['font.family'] = 'serif'
        matplotlib.rcParams['font.serif'] = ['Palatino Linotype']
        plt.figure(figsize=(8, 6))
        plt.subplot(2, 2, 1, axis_bgcolor=VIRIDIS_PURPLE)
        plt.xlim((-1, 3))
        plt.hexbin(w1_w2[pos], probs[pos], cmap='viridis', gridsize=30,
                   bins='log', linewidths=0.5)
        plt.colorbar()
        plt.xlabel('$w1 - w2$')
        plt.ylabel('Predicted probability')
        plt.title('z = 1')
        plt.subplot(2, 2, 2, axis_bgcolor=VIRIDIS_PURPLE)
        plt.xlim((-1, 3))
        plt.hexbin(w1_w2[neg], probs[neg], cmap='viridis', gridsize=30,
                   bins='log', linewidths=0.5)
        plt.title('z = 0')
        plt.colorbar()
        plt.xlabel('$w1 - w2$')
        plt.ylabel('Predicted probability')
        plt.subplot(2, 2, 3, axis_bgcolor=VIRIDIS_PURPLE)
        plt.hexbin(w2_w3[pos], probs[pos], cmap='viridis', gridsize=30,
                   bins='log', linewidths=0.5)
        plt.xlim((-0.5, 5.5))
        plt.colorbar()
        plt.xlabel('$w2 - w3$')
        plt.ylabel('Predicted probability')
        plt.subplot(2, 2, 4, axis_bgcolor=VIRIDIS_PURPLE)
        plt.hexbin(w2_w3[neg], probs[neg], cmap='viridis', gridsize=30,
                   bins='log', linewidths=0.5)
        plt.colorbar()
        plt.xlim((-0.5, 5.5))
        plt.xlabel('$w2 - w3$')
        plt.ylabel('Predicted probability')
        plt.subplots_adjust(wspace=0.32, hspace=0.32)
        plt.show()

        # gridsize = 100

        # xs_log = numpy.linspace(-1, 6, gridsize)
        # xs = numpy.power(10, xs_log)
        # ys_log = numpy.linspace(-1, 6, gridsize)
        # ys = numpy.power(10, ys_log)
        # xs_, ys_ = numpy.meshgrid(xs, ys)
        # features_ = numpy.zeros((gridsize ** 2, features.shape[1]))
        # features_[:, 4] = ys_.ravel()
        # features_[:, 5] = xs_.ravel()
        # probs = lr.predict_proba(features_)[:, 1]

        # plt.pcolormesh(xs_log, ys_log, probs.reshape((gridsize, gridsize)),
        #                cmap='viridis')
        # plt.show()

        # # plt.subplot(2, 2, 1, axis_bgcolor=VIRIDIS_PURPLE)
        # # plt.title('Crowd positives')
        # # plt.hexbin(w2_w3[labels == 1], w1_w2[labels == 1], gridsize=gridsize,
        # #            cmap='viridis', bins=None)
        # # plt.xlim((-1, 6))
        # # plt.ylim((-1, 3))
        # # plt.subplot(2, 2, 2, axis_bgcolor=VIRIDIS_PURPLE)
        # # plt.title('Crowd negatives')
        # # plt.hexbin(w2_w3[labels == 0], w1_w2[labels == 0], gridsize=gridsize,
        # #            cmap='viridis', bins=None)
        # # plt.xlim((-1, 6))
        # # plt.ylim((-1, 3))
        # # plt.subplot(2, 2, 3, axis_bgcolor=VIRIDIS_PURPLE)
        # # plt.title('p(z) > 0.5')
        # # plt.hexbin(w2_w3[pos], w1_w2[pos], gridsize=gridsize, cmap='viridis',
        # #            bins=None)
        # # plt.xlim((-1, 6))
        # # plt.ylim((-1, 3))
        # # plt.subplot(2, 2, 4, axis_bgcolor=VIRIDIS_PURPLE)
        # # plt.title('p(z) < 0.5')
        # # plt.hexbin(w2_w3[neg], w1_w2[neg], gridsize=gridsize, cmap='viridis',
        # #            bins=None)
        # # plt.xlim((-1, 6))
        # # plt.ylim((-1, 3))
        # # plt.show()

main('../../data/training.h5')
