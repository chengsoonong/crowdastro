"""Object for storing experimental results.

Matthew Alger
The Australian National University
2016
"""

import h5py
import numpy


class Results(object):
    """Stores experimental results."""

    def __init__(self, path, methods, n_splits, n_examples):
        """
        path: Path to the results file (h5). File will be created if it does not
            already exist.
        methods: List of methods being tested.
        n_splits: Number of data splits.
        n_examples: Number of examples.
        """
        if not path.endswith('.h5'):
            path += '.h5'
        self.h5_path = path
        self.methods = {method:index for index, method in enumerate(methods)}
        self.n_methods = len(methods)
        self.n_splits = n_splits
        self.n_examples = n_examples

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
    
    def store_result(self, method, split, example, result):
        """Stores a single result.

        method: Method. str
        split: Split ID. int
        example: Example ID. int
        result: Result to store. float
        """
        with h5py.File(self.h5_path, 'r+') as f:
            f['results'][self.methods[method], split, example] = result

    def store_trial(self, method, split, results):
        """Stores results from one trial.

        method: Method ID. int
        split: Split ID. int
        results: Results for each example. (n_examples,) array
        """
        with h5py.File(self.h5_path, 'r+') as f:
            f['results'][self.methods[method], split] = results

    def __getitem__(self, *args, **kwargs):
        with h5py.File(self.h5_path, 'r+') as f:
            try:
                args = (self.methods[args[0]],) + args[1:]
            except TypeError:
                pass

            return f['results'].__getitem__(*args, **kwargs)

    def __repr__(self):
        return 'Results({}, methods={}, n_splits={}, n_examples={})'.format(
                repr(self.h5_path),
                sorted(self.methods, key=self.methods.get),
                self.n_splits, self.n_examples)
