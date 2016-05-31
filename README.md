# crowdastro
Machine learning using crowd sourced data in astronomy.

For setup details, see the documentation [here](docs/setup.md), in particular start
[mongodb](https://www.mongodb.com/).
```bash
mongod --config /usr/local/etc/mongod.conf --fork
```

For a brief description of each notebook, see the documentation [here](docs/notebooks.md).

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
python3 -m crowdastro training_data processed.db consensuses gator_cache training.h5 --atlas
```

Preprocess the CNN inputs/outputs:

```bash
python3 -m crowdastro.preprocess_cnn_images training.h5 data
```

Here, `data` is the directory containing the `cdfs` and `elais` directories.

Generate a model:

```bash
python3 -m crowdastro.compile_cnn model.json
```

Train the CNN:

```bash
python3 -m crowdastro.train_cnn training.h5 model.json weights.h5 n_epochs batch_size
```

Coming soon: Fitting PCA, and finally classifying.

## Generating the Radio Galaxy Zoo catalogue

Process the consensuses as above, then generate the catalogue:

```bash
python -m crowdastro catalogue processed.db consensuses gator_cache hosts radio_components --atlas
```
