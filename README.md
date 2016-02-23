# crowdastro
Machine learning using crowd sourced data in astronomy

# Setup

Install MongoDB. Restore the database:

```bash
mongorestore --db radio --drop --collection\
  radio_subjects radio/radio_subjects.bson
mongorestore --db radio --drop --collection\
  radio_classifications radio/radio_classifications.bson
```

