# Notebooks

<!--0. mongo2pandas &mdash; Quick example of loading MongoDB data into Pandas.-->
1. simple_rgz &mdash; Basic manipulation of Radio Galaxy Zoo data.
2. potential_host_counting &mdash; Investigating detecting potential hosts galaxies in infrared images, and counting how many there are in each subject.
3. click_consensus &mdash; Computing the radio consensus and click consensus with a PG-means Gaussian mixture algorithm, attempting to improve robustness and efficiency upon the existing method (Banfield et al., 2015).
4. fits_format &mdash; Loading images from a local database of FITS images.
5. training_data &mdash; Assembling a set of training inputs and targets, then running these through logistic regression and a convolutional neural network.
6. feature_extraction &mdash; Attempting to use a convolutional autoencoder to learn features of IR images. (Incomplete)
7. mean_images &mdash; Computing the mean IR and radio image around potential hosts to see if there is a difference in the IR image between hosts and non-hosts (there isn't).
8. click_consensus_distribution &mdash; Attempting to find out whether majority vote is a good way to decide what the collective classification made by the RGZ volunteers was (it is).
9. pipeline &mdash; End-to-end classification pipeline.
10. potential_host_detection &mdash; Detecting potential hosts with scikit-image.
11. classification &mdash; Trying different classifiers.
12. labels &mdash; Ironing out the label processing pipeline so we can freeze the labels.
13. source_catalogue &mdash; Investigating using astronomical catalogues to find potential hosts and features.
14. cnn &mdash; Training a convolutional neural network to learn features of radio images.
15. cnn_test &mdash; Running the CNN from notebook 14.
16. pca &mdash; Principal component analysis on the CNN outputs.
17. training_with_features &mdash; Running the whole pipeline through logistic regression.
18. distance_from_centre &mdash; Testing whether the distance from the centre of the image is a useful feature.
