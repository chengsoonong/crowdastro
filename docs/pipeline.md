# Pipeline

The processing pipeline is divided up into a number of scripts, each of which outputs into an SQLite3 database. They need to be run in order, but once you have the database output from each step, you can start from that step in future.

## Parsing raw classifications

Reads the raw Radio Galaxy Zoo classifications into a simpler format and does some data validation. All invalid data points are discarded.

```
python -m crowdastro raw_classifications database.db classifications
```
