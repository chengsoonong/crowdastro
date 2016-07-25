# Crowdastro Dataset

[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.58316.svg)](http://dx.doi.org/10.5281/zenodo.58316)

The Crowdastro dataset is a set of training data for the binary classification problem of determining whether a galaxy hosts a radio source. The dataset contains features and labels for all 24140 objects detected in the WISE infrared survey that are within 1 arcminute of an object detected in the ATLAS radio survey. The prediction task is to predict the label of an object given its features.

The features are not scaled and have not undergone any feature extraction process. The first four features are the fluxes of the object in different WISE bands. The fifth feature is the distance to the nearest ATLAS radio object. The final 1024 features are a 0.8 arcminute patch of radio sky centred on the object.

The labels are based on the consensus locations from the Radio Galaxy Zoo, matched to the nearest WISE object. WISE objects matched to a consensus location have the label 1, and all other objects have the label 0. Consensuses are found by collating volunteer clicks from the Radio Galaxy Zoo and fitting a Gaussian mixture model. The mean of the Gaussian with the highest total membership is the consensus location. The number of Gaussians is found by a grid search minimising the Baysian information criterion.

The dataset file is an HDF5 file containing two tables named "features" and "labels". Each row of the features table is the features representing one object. The ith element of the labels table is the label corresponding to the ith row of the features table.

Code to generate the dataset is [on GitHub](https://github.com/chengsoonong/crowdastro). The dataset itself is available [on Zenodo](http://dx.doi.org/10.5281/zenodo.58316).

This dataset makes use of data products from the Wide-field Infrared Survey Explorer, which is a joint project of the University of California, Los Angeles, and the Jet Propulsion Laboratory/California Institute of Technology, funded by the National Aeronautics and Space Administration.

This dataset includes data from the Australia Telescope Compact Array. Australia Telescope Compact Array is part of the Australia Telescope, which is funded by the Commonwealth of Australia for operation as a National Facility managed by CSIRO.

This dataset includes data from the Radio Galaxy Zoo. The Radio Galaxy Zoo data has been made possible by the participation of more then 10,000 volunteers in the Radio Galaxy Zoo project. Their contributions are individually acknowledged at http://rgzauthors.galaxyzoo.org.
