# crowdastro

This project aims to develop a machine learned method for cross-identifying radio objects and their host galaxies, using crowdsourced labels from the [Radio Galaxy Zoo](http://radio.galaxyzoo.org).

For setup details, see the documentation [here](docs/setup.md).

For a brief description of each notebook, see the documentation [here](docs/notebooks.md).

The cross-identification dataset is available [on Zenodo](http://dx.doi.org/10.5281/zenodo.58316).

[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.58316.svg)](http://dx.doi.org/10.5281/zenodo.58316)

## Running the pipeline

Import data sources:

```bash
python3 -m crowdastro.import_data
```

Process the consensuses:

```bash
python3 -m crowdastro.consensuses
```

Generate the training data:

```bash
python3 -m crowdastro.generate_training_data
```

The training data contains `features` and `labels`. Note that image features have not been processed &mdash; these are raw pixels.

Generate a model:

```bash
python3 -m crowdastro.compile_cnn
```

Train the CNN:

```bash
python3 -m crowdastro.train_cnn
```

Generate the CNN outputs:

```bash
python3 -m crowdastro.generate_cnn_outputs
```

Note that this mutates the `features` dataset, replacing image features with the CNN outputs.

Repack the H5 file:

```bash
python3 crowdastro.repack_h5 training.h5
```

Train a classifier:

```bash
python3 -m crowdastro.train --classifier lr
```

Test the classifier against subjects:

```bash
python3 -m crowdastro.test
```

## Generating the Radio Galaxy Zoo catalogue

Process the consensuses as above, then generate the catalogue:

```bash
python -m crowdastro catalogue processed.db consensuses gator_cache hosts radio_components --atlas
```
