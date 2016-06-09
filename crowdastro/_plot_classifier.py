import argparse
import logging

import h5py
import matplotlib
import matplotlib.pyplot as plot
import numpy
import sklearn.externals
import sklearn.metrics


font = {'family' : 'Palatino Linotype',
        'size'   : 22}

matplotlib.rc('font', **font)


def main(inputs_h5, training_h5, classifier_path, astro_transformer_path,
         image_transformer_path, use_astro=True, use_cnn=True, simple=False):
    classifier = sklearn.externals.joblib.load(classifier_path)
    astro_transformer = sklearn.externals.joblib.load(astro_transformer_path)
    image_transformer = sklearn.externals.joblib.load(image_transformer_path)

    testing_indices = inputs_h5['/atlas/cdfs/testing_indices'].value
    all_astro_inputs = astro_transformer.transform(training_h5['astro'].value)
    all_cnn_inputs = image_transformer.transform(
            training_h5['cnn_outputs'].value)
    all_inputs = numpy.hstack([all_astro_inputs, all_cnn_inputs])
    all_labels = training_h5['labels'].value
    
    inputs = all_inputs[testing_indices]
    labels = all_labels[testing_indices]
    probs = classifier.predict_proba(inputs)
    outputs = classifier.predict(inputs)
    
    precision, recall, _ = sklearn.metrics.precision_recall_curve(labels,
                                                                  probs[:, 1])
    plot.figure(figsize=(20, 10))
    plot.subplot(1, 2, 1)
    plot.plot(recall, precision)
    plot.xlabel('Recall')
    plot.ylabel('Precision')

    plot.subplot(1, 2, 2)
    fpr, tpr, _ = sklearn.metrics.roc_curve(labels, probs[:, 1])
    plot.plot(fpr, tpr)
    plot.xlabel('False positive rate')
    plot.ylabel('True positive rate')
    plot.show()

    cm = sklearn.metrics.confusion_matrix(labels, outputs)
    cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
    plot.imshow(cm, interpolation='nearest', cmap=plot.cm.Blues)
    plot.colorbar()
    tick_marks = numpy.arange(2)
    plot.xticks(tick_marks, ['not host', 'host'], rotation=45)
    plot.yticks(tick_marks, ['not host', 'host'])
    plot.tight_layout()
    plot.ylabel('True label')
    plot.xlabel('Predicted label')
    plot.show()
    
    print('Accuracy: {:.02%}'.format(classifier.score(inputs, labels)))

    n_true_pos = numpy.sum(numpy.logical_and(labels == outputs, labels == 1))
    n_true_neg = numpy.sum(numpy.logical_and(labels == outputs, labels == 0))
    n_false_pos = numpy.sum(numpy.logical_and(labels != outputs, labels == 0))
    n_false_neg = numpy.sum(numpy.logical_and(labels != outputs, labels == 1))
    print('Balanced accuracy: {:.02%}'.format(0.5 * n_true_pos / (n_true_pos + n_false_neg) + 0.5 * n_true_neg / (n_true_neg + n_false_pos)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', default='crowdastro.h5',
                        help='HDF5 inputs data file')
    parser.add_argument('--training', default='training.h5',
                        help='HDF5 training data file')
    parser.add_argument('--classifier', default='classifier.pkl',
                        help='classifier file')
    parser.add_argument('--astro_transformer', default='astro_transformer.pkl',
                        help='astro transformer file')
    parser.add_argument('--image_transformer', default='image_transformer.pkl',
                        help='image transformer file')
    parser.add_argument('--no_astro', action='store_false', default=True,
                        help='ignore astro features')
    parser.add_argument('--no_cnn', action='store_false', default=True,
                        help='ignore CNN features')
    parser.add_argument('--simple', action='store_true', default=False,
                        help='use only single-AGN subjects')
    args = parser.parse_args()

    logging.root.setLevel(logging.DEBUG)

    with h5py.File(args.training, 'r') as training_h5:
        with h5py.File(args.inputs, 'r') as inputs_h5:
            main(inputs_h5, training_h5, args.classifier,
                 args.astro_transformer, args.image_transformer,
                 use_astro=args.no_astro, use_cnn=args.no_cnn,
                 simple=args.simple)
