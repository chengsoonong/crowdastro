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

You need the following four data files:

- ATLASDR3_cmpcat_23July2015.dat.gz
- cdfs_11JAN2014_2x2_5x5.tgz
- elais_11JAN2014_2x2_5x5.tgz
- sanitized_radio_2016-03-01.tar.gz

These files are currently not easily available. For access, email the [Radio Galaxy Zoo team](https://github.com/zooniverse/Radio-Galaxy-Zoo).

From the project root, run setup_data.sh. If this doesn't work, extract sanitized_radio_2016-03-01.tar.gz and ATLASDR3_cmpcat_23July2015.dat.gz to data/, elais_11JAN2014_2x2_5x5.tgz to data/elais, and cdfs_11JAN2014_2x2_5x5.tgz to data/cdfs.

Run mongo_load_json.sh to import the files into the `radio` MongoDB database. You will now be able to access the collections `radio.radio_classifications`, `radio.radio_groups`, and `radio.radio_subjects`.
