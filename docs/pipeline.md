# Running the Pipeline

Import data sources:

```bash
crowdastro import_data
```
This by default uses the SWIRE catalogue. For using the WISE catalogue:
```bash
crowdastro import_data --ir wise
```

Process the consensuses:

```bash
crowdastro consensuses
```

Generate the training data:

```bash
crowdastro generate_training_data
```

The training data contains `raw_features` and `labels`. Note that image features have not been processed &mdash; these are raw pixels.

Generate a model:

```bash
crowdastro compile_cnn
```

Train the CNN:

```bash
crowdastro train_cnn
```

Generate the CNN outputs:

```bash
crowdastro generate_cnn_outputs
```

This adds the `features` dataset to the training HDF5 file.

Train a classifier:

```bash
crowdastro train --classifier lr
```

Test the classifier against subjects:

```bash
crowdastro test
```
