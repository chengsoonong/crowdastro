# crowdastro
Machine learning using crowd sourced data in astronomy.

For setup details, see the documentation [here](docs/setup.md).

For a brief description of each notebook, see the documentation [here](docs/notebooks.md).

## Running the pipeline

Process the raw classifications:

```bash
python3 -m crowdastro raw_classifications processed.db classifications --atlas
```

Process the consensuses:

```bash
python3 -m crowdastro consensuses processed.db classifications consensuses --atlas
```

Generate the training data:

```bash
python3 -m crowdastro training_data processed.db consensuses gator_cache training.h5 --atlas
```

Preprocess the CNN inputs/outputs:

```bash
crowdastro/preprocess_cnn_images.py training.h5 data
```

Here, `data` is the directory containing the `cdfs` and `elais` directories.

Generate a model:

```bash
crowdastro/compile_cnn.py model.json
```

Train the CNN:

```bash
crowdastro/train_cnn.py training.h5 model.json weights.h5 n_epochs batch_size
```

Coming soon: Fitting PCA, and finally classifying.

## Generating the Radio Galaxy Zoo catalogue

Process the consensuses as above, then generate the catalogue:

```bash
python -m crowdastro catalogue processed.db consensuses gator_cache hosts radio_components --atlas
```
