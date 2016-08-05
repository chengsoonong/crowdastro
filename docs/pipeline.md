# Running the Pipeline

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
python3 -m crowdastro.repack_h5 training.h5
```

Train a classifier:

```bash
python3 -m crowdastro.train --classifier lr
```

Test the classifier against subjects:

```bash
python3 -m crowdastro.test
```
