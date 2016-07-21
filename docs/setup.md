# Setup

## Python

To install crowdastro and dependencies, you need Python, pip, and git.

```bash
git clone https://github.com/chengsoonong/crowdastro.git && cd crowdastro
pip install -r requirements.txt
```

## MongoDB

Install and run MongoDB. On Mac OS X or Linux, you can run MongoDB as follows:

```bash
mongod --config /usr/local/etc/mongod.conf --fork
```

On Windows:
```batch
K:\MongoDB\Server\3.2\bin\mongod.exe --dbpath J:\data\mongodb
```

## Data

You need the following six data files:

- ATLASDR3_cmpcat_23July2015.dat.gz
- cdfs_11JAN2014_2x2_5x5.tgz
- sanitized_radio_2016-03-01.tar.gz
- RGZ-ATLAS-Bookkeeping.csv
- [SWIRE3_CDFS_cat_IRAC24_21Dec05.tbl.gz](http://swire.ipac.caltech.edu/swire/astronomers/data/SWIRE3_CDFS_cat_IRAC24_21Dec05.tbl.gz) or an AllWISE catalogue covering the CDFS field
- norris_2006_atlas_classifications_ra_dec_only.dat (pipe-separated RA/DEC from Norris et al. (2006) Table 6)

These files are currently not easily available. For access, email the [Radio Galaxy Zoo team](https://github.com/zooniverse/Radio-Galaxy-Zoo).

The following steps assume the files are in the `data` subdiresctory. If this isn't the case, modify paths in `setup_data.sh`, `mongo_load_json.sh`, and `crowdastro.json`.

From the project root, run setup_data.sh. If this doesn't work, extract sanitized_radio_2016-03-01.tar.gz, SWIRE3_CDFS_cat_IRAC24_21Dec05.tbl.gz, and ATLASDR3_cmpcat_23July2015.dat.gz to data/, and cdfs_11JAN2014_2x2_5x5.tgz to data/cdfs. Copy RGZ-ATLAS-Bookkeeping.csv and any other files to data/. These paths can all be modified in crowdastro.json in the root directory.

Run mongo_load_json.sh to import the files into the `radio` MongoDB database. You will now be able to access the collections `radio.radio_classifications`, `radio.radio_groups`, and `radio.radio_subjects`.
