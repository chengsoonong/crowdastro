"""Object for storing experimental results.

Matthew Alger
The Australian National University
2016
"""

import h5py
import numpy


class Results(object):
    """Stores experimental results."""

    def __init__(self, path, methods, n_splits, n_examples, n_params):
        """
        path: Path to the results file (h5). File will be created if it does not
            already exist.
        methods: List of methods being tested.
        n_splits: Number of data splits.
        n_examples: Number of examples.
        n_params: Number of parameters in the model.
        """
        if not path.endswith('.h5'):
            path += '.h5'
        self.h5_path = path
        self.methods = {method:index for index, method in enumerate(methods)}
        self.n_methods = len(methods)
        self.n_splits = n_splits
        self.n_examples = n_examples
        self.n_params = n_params

        try:
            with h5py.File(path, 'r+') as f:
                self._create(f)
        except OSError:
            with h5py.File(path, 'w') as f:
                self._create(f)

        # We could store a reference to the actual file, but then there's no
        # guarantee we'll close it safely later.

    def _create(self, f):
        """Creates the results dataset."""
        if 'results' not in f:
            f.create_dataset('results',
                    shape=(self.n_methods, self.n_splits, self.n_examples))
        if 'models' not in f:
            f.create_dataset('models',
                    shape=(self.n_methods, self.n_splits, self.n_params))
        if 'run_flag' not in f:
            f.create_dataset('run_flag',
                    shape=(self.n_methods, self.n_splits, self.n_examples),
                    data=numpy.zeros(
                            (self.n_methods, self.n_splits, self.n_examples)))

    def store_trial(self, method, split, results, params, indices=None):
        """Stores results from one trial.

        method: Method ID. int
        split: Split ID. int
        results: Results for each example. (n_examples,) array
        params: (n_params,) array representing the classifier.
        indices: Indices of examples in this trial. [int]. Default all indices.
        """
        with h5py.File(self.h5_path, 'r+') as f:
            if indices is not None:
                f['results'][self.methods[method], split, indices] = results
                f['run_flag'][self.methods[method], split, indices] = 1
            else:
                f['results'][self.methods[method], split] = results
                f['run_flag'][self.methods[method], split] = 1
            f['models'][self.methods[method], split] = params

    @property
    def models(self):
        with h5py.File(self.h5_path, 'r') as f:
            return f['models'].value

    def __getitem__(self, *args, **kwargs):
        with h5py.File(self.h5_path, 'r') as f:
            try:
                args = (self.methods[args[0]],) + args[1:]
            except TypeError:
                pass

            return f['results'].__getitem__(*args, **kwargs)

    def __repr__(self):
        return 'Results({}, methods={}, n_splits={}, n_examples={}, ' \
                        'n_params={})'.format(
                repr(self.h5_path),
                sorted(self.methods, key=self.methods.get),
                self.n_splits, self.n_examples, self.n_params)
