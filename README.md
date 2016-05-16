# crowdastro
Machine learning using crowd sourced data in astronomy.

For setup details, see the documentation [here](docs/setup.md).

For a brief description of each notebook, see the documentation [here](docs/notebooks.md).

## Running the pipeline

Process the raw classifications:

```bash
python -m crowdastro raw_classifications processed.db classifications --atlas
```

Process the consensuses:

```bash
python -m crowdastro consensuses processed.db classifications consensuses --atlas
```

Generate the training data:

```bash
python -m crowdastro training_data processed.db consensuses gator_cache training.h5 --atlas
```

Coming soon: Training a CNN and PCA, and finally classifying.

## Generating the Radio Galaxy Zoo catalogue

Process the consensuses as above, then generate the catalogue:

```bash
python -m crowdastro catalogue processed.db consensuses gator_cache hosts radio_components --atlas
```
