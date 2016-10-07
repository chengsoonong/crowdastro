import argparse

import h5py
import sklearn.externals.joblib
import sklearn.linear_model


def main(training_h5_path, classifier_path):
    with h5py.File(training_h5_path, 'r') as training_h5:
        features = training_h5['features']
        labels = training_h5['labels']
        lr = sklearn.linear_model.LogisticRegression(class_weight='balanced')
        lr.fit(features, labels)
        sklearn.externals.joblib.dump(lr, classifier_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training')
    parser.add_argument('--classifier')
    args = parser.parse_args()
    main(args.training, args.classifier)

