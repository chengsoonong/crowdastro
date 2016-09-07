"""Object for storing experimental results.

Matthew Alger
The Australian National University
2016
"""

import json

import h5py
import numpy


class Results(object):
    """Stores experimental results."""

    def __init__(self, path, methods, n_splits, n_examples, n_params, model):
        """
        path: Path to the results file (h5). File will be created if it does not
            already exist.
        methods: List of methods being tested.
        n_splits: Number of data splits.
        n_examples: Number of examples.
        n_params: Number of parameters in the model.
        model: String representing the model function and version.
        """
        if not path.endswith('.h5'):
            path += '.h5'
        self.h5_path = path
        self.methods = methods
        self.n_methods = None  # Initialised later.
        self.n_splits = n_splits
        self.n_examples = n_examples
        self.n_params = n_params
        self.model = model

        try:
            with h5py.File(path, 'r+') as f:
                self._create(f)
        except OSError:
            with h5py.File(path, 'w') as f:
                self._create(f)

        # We could store a reference to the actual file, but then there's no
        # guarantee we'll close it safely later.

    @classmethod
    def from_path(cls, path):
        """
        Loads a Results object from a path.
        """
        if not path.endswith('.h5'):
            path += '.h5'
        with h5py.File(path, 'r') as f:
            methods = json.loads(f.attrs['methods'])
            n_splits = f['results'].shape[1]
            n_examples = f['results'].shape[2]
            assert len(methods) == f['results'].shape[0]
            n_params = f['models'].shape[2]
            model = f.attrs['model']

        return cls(path, methods, n_splits, n_examples, n_params, model)

    def _create(self, f):
        """Creates the results dataset."""
        if 'methods' in f.attrs:
            # What methods are we adding and what methods do we already have?
            existing_methods = json.loads(f.attrs['methods'])
            new_methods = [m for m in self.methods
                           if m not in existing_methods]
            f.attrs['methods'] = json.dumps(existing_methods + new_methods)
            self.methods = existing_methods + new_methods
            n_existing_methods = len(existing_methods)
        else:
            f.attrs['methods'] = json.dumps(self.methods)
            n_existing_methods = 0

        self.method_idx = {j:i for i, j in enumerate(self.methods)}
        self.n_methods = len(self.method_idx)

        results_shape = (self.n_methods, self.n_splits, self.n_examples)
        if 'results' in f and f['results'].shape[0] != self.n_methods:
            # Extend the HDF5 file to hold new methods.
            f.create_dataset('results_', data=f['results'].value)
            del f['results']
            f.create_dataset('results', shape=results_shape, dtype=float)
            f['results'][:n_existing_methods] = f['results_'].value
            del f['results_']
        elif 'results' not in f:
            f.create_dataset('results', shape=results_shape, dtype=float)
        else:
            assert f['results'].shape == results_shape, \
                    'results: Expected shape {}, found {}.'.format(
                            results_shape, f['results'].shape)

        models_shape = (self.n_methods, self.n_splits, self.n_params)
        if 'models' in f and f['models'].shape[0] != self.n_methods:
            # Extend the HDF5 file to hold new methods.
            f.create_dataset('models_', data=f['models'].value)
            del f['models']
            f.create_dataset('models', shape=models_shape, dtype=float)
            f['models'][:n_existing_methods] = f['models_'].value
            del f['models_']
        elif 'models' not in f:
            f.create_dataset('models', shape=models_shape, dtype=float)
            f.attrs['model'] = self.model
            assert f.attrs['model'] == self.model
        else:
            assert f['models'].shape == models_shape, \
                    'models: Expected shape {}, found {}.'.format(
                            models_shape, f['models'].shape)

        run_flag_shape = (self.n_methods, self.n_splits, self.n_examples)
        if 'run_flag' in f and f['run_flag'].shape[0] != self.n_methods:
            # Extend the HDF5 file to hold new methods.
            f.create_dataset('run_flag_', data=f['run_flag'].value)
            del f['run_flag']
            f.create_dataset('run_flag', data=numpy.zeros(run_flag_shape))
            f['run_flag'][:n_existing_methods] = f['run_flag_'].value
            del f['run_flag_']
        elif 'run_flag' not in f:
            f.create_dataset('run_flag', shape=run_flag_shape,
                             data=numpy.zeros(run_flag_shape))
        else:
            assert f['run_flag'].shape == run_flag_shape, \
                    'run_flag: Expected shape {}, found {}.'.format(
                            run_flag_shape, f['run_flag'].shape)


    def store_trial(self, method, split, results, params, indices=None):
        """Stores results from one trial.

        method: Method. str
        split: Split ID. int
        results: Results for each example. (n_examples,) array
        params: (n_params,) array representing the classifier.
        indices: Indices of examples in this trial. [int]. Default all indices.
        """
        with h5py.File(self.h5_path, 'r+') as f:
            if indices is not None:
                f['results'][self.method_idx[method], split, indices] = results
                f['run_flag'][self.method_idx[method], split, indices] = 1
            else:
                f['results'][self.method_idx[method], split] = results
                f['run_flag'][self.method_idx[method], split] = 1
            f['models'][self.method_idx[method],
                    split, :params.shape[0]] = params

    def has_run(self, method, split):
        """Returns whether a trial has run successfully.

        If *any* example in the trial has been run successfully, then the trial
        has run successfully.

        method: Method. str.
        split: Split ID. int
        -> bool
        """
        with h5py.File(self.h5_path, 'r') as f:
            return any(
                    f['run_flag'][self.method_idx[method], split].astype(bool))

    @property
    def models(self):
        with h5py.File(self.h5_path, 'r') as f:
            return f['models'].value

    def get_mask(self, method, split):
        """Get a mask for trials that have been run successfully.

        Mask will be 1 for trials that have run, and 0 otherwise.

        method: Method. str
        split: Split ID. int
        -> (n_examples,) array
        """
        with h5py.File(self.h5_path, 'r') as f:
            return f['run_flag'][self.method_idx[method], split].astype(bool)

    def __getitem__(self, item, *args, **kwargs):
        with h5py.File(self.h5_path, 'r') as f:
            try:
                item = (self.method_idx[item[0]],) + item[1:]
            except TypeError as e:
                print(item)
                print(e)
                pass

            return f['results'].__getitem__(item, *args, **kwargs)

    def __repr__(self):
        return 'Results({}, methods={}, n_splits={}, n_examples={}, ' \
                        'n_params={})'.format(
                repr(self.h5_path),
                sorted(self.method_idx, key=self.method_idx.get),
                self.n_splits, self.n_examples, self.n_params)
