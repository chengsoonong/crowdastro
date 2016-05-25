# Notes for extracting the annotations from citizen scientists

## MongoDB setup

```
brew install mongodb
```

```
mongod --config /usr/local/etc/mongod.conf --fork
```

## Restoring the database

```
mongorestore --db radio --drop --collection\
  radio_users anonymized/radio_users.bson
mongorestore --db radio --drop --collection\
  radio_subjects anonymized/radio_subjects.bson
mongorestore --db radio --drop --collection\
  radio_classifications anonymized/radio_classifications.bson
```

There's a description of the database schema [online](https://github.com/willettk/rgz-analysis/blob/master/RadioGalaxyZoo_datadescription.ipynb), along with some examples of how to use Python to access Mongo programmatically.

## Looking at the annotations with PyMongo

```python
import pprint
from pymongo import MongoClient
from IPython.display import Image
client = MongoClient('localhost', 27017)
db = client['radio']
```

Database #1: Radio subjects

```python
subjects = db['radio_subjects']
sample_subject = subjects.find_one()
pprint.pprint(sample_subject)
# Radio image
Image(url=sample_subject['location']['radio'])
# Infrared image
Image(url=sample_subject['location']['standard'])
```

Database #2: Radio classifications

```python
classifications = db['radio_classifications']
my_id = db['radio_users'].find_one({'name':'KWillett'})['_id']
sample_classification = classifications.find_one({'user_id':my_id})
pprint.pprint(sample_classification)
```

Database #3: Radio users

```python
users = db['radio_users']
sample_user = users.find_one({'name': 'KWillett'})
pprint.pprint(sample_user)
```


## Installing Monary to extract annotations to Pandas

Make sure you have OpenSSL
```
brew install openssl
export LDFLAGS="-L/usr/local/opt/openssl/lib"
export CPPFLAGS="-I/usr/local/opt/openssl/include"
```

Download mongodb C driver (brew also has mongo-c, which might work)
```
curl -LO https://github.com/mongodb/mongo-c-driver/releases/download/1.2.1/mongo-c-driver-1.2.1.tar.gz
tar xzf mongo-c-driver-1.2.1.tar.gz
cd mongo-c-driver-1.2.1
```

Build and install with SSL
```
./configure --enable-ssl=yes --libdir=/usr/lib
make
sudo make install
```

Install monary. Need to force OpenSSL to link to /usr/lib/libcrypto.dylib
```
brew link openssl --force
pip install monary
```
