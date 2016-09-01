# Running the Pipeline

Import data sources:

```bash
crowdastro import_data
```

Process the consensuses:

```bash
crowdastro consensuses
```

Generate the training data:

```bash
crowdastro generate_training_data
```

The training data contains `features` and `labels`. Note that image features have not been processed &mdash; these are raw pixels.

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

Note that this mutates the `features` dataset, replacing image features with the CNN outputs.

Repack the H5 file:

```bash
crowdastro repack_h5 training.h5
```

Train a classifier:

```bash
crowdastro train --classifier lr
```

Test the classifier against subjects:

```bash
crowdastro test
```
