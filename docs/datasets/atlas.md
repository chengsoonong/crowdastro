# Australian Telescope Large Area Survey (ATLAS)

A detailed description of the ATLAS survey can be found in [Franzen et al. (2015)](http://adsabs.harvard.edu/abs/2015MNRAS.453.4020F).  There are two ATLAS fields: (1) Chandra Deep Field South (CDFS); and (2) European Large Area ISO survey South 1 (ELAIS S1).

## Field data

The data for both fields are contained in one ASCII file (ATLASDR3_cmpcat_23July2013.dat).  The key columns for this project are:

| Column | Description                                                                                                       |
|--------|-------------------------------------------------------------------------------------------------------------------|
| 1      | ID: radio source identifier (NOTE: this will be different from the FITS images - results from timing of projects) |
| 3      | RA: right ascension of radio subject in hh:mm:ss.ss                                                               |
| 4      | DEC: declination of radio subject in dd:mm:ss.s                                                                   |
| 9      | rms: the background level of the radio image.,This value was used to calculate the contours in the RGZ overlay.   |
| 12     | Sp: peak flux of the radio subject - or the island                                                                |

In the ID column you will see that there may be some names with a C# after them (i.e., EI0011C1 and EI0011C2).  These have been classified by astronomers to be components of one radio subject with one host galaxy.

## Images

There are two tarballs containing the FITS images and metadata file corresponding to these images.  The are labelled cdfs_11JAN2014_2x2_5x5.tgz and elais_11JAN2014_2x2_5x5.tgz .  The files are split into two directories 2x2 and 5x5.  These contain the images for a 2x2 arcminute postage stamp and 5x5 arcminute postage stamp.  In these directories you will find 5 files for each radio subject:

1. ID_heatmap.png :  background heatmap image (infrared image) centred on the radio subject.

2. ID_heatmap+contours.png : the image you see on radio galaxy zoo with the radio contours overlaid centred on the radio subject.

3. ID_ir.fits : the background FITS image file (infrared data) centred on the radio subject.

4. ID_radio.fits : the radio FITS image centred on the radio subject.

5. ID_radio.png : the radio image centred on the radio subject.

In addition you will find a file labelled FIELD-input_cat_11JAN2014_metadata.txt . This file contains all the information on the files sent through to be placed on Radio Galaxy Zoo.  The final column in this file is the background level in the radio FITS image that was used to calculate the contours.

## ELAIS and CDFS radio classifications

The ATLAS classifications for a set of radio subjects was completed by astronomers (CDFS: [Norris et al. (2006)](http://adsabs.harvard.edu/abs/2006AJ....132.2409N) and  ELAIS S1: [Middelberg et al. (2008)](http://adsabs.harvard.edu/abs/2008AJ....135.1276M)).  The catalogues for these identifications are found here:

* [CDFS catalogue](http://vizier.cfa.harvard.edu/viz-bin/VizieR?-source=J/AJ/132/2409)
* [ELAIS S1 catalogue](http://vizier.cfa.harvard.edu/viz-bin/VizieR?-source=J/AJ/135/1276)

## Infrared catalgoue

The infrared data for ATLAS comes from SWIRE.  The catalogues are found here: [http://swire.ipac.caltech.edu/swire/astronomers/data_access.html](http://swire.ipac.caltech.edu/swire/astronomers/data_access.html)
