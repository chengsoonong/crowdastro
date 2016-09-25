mkdir data
tar -zxvf radio_galaxy_zoo_2016-08-28.tar.gz -C data
mv data/radio_galaxy_zoo_2016-08-28/* data/
rmdir data/radio_galaxy_zoo_2016-08-28
gunzip -c ATLASDR3_cmpcat_23July2015.dat.gz > data/ATLASDR3_cmpcat_23July2015.dat
gunzip -c SWIRE3_CDFS_cat_IRAC24_21Dec05.tbl.gz > data/SWIRE3_CDFS_cat_IRAC24_21Dec05.tbl
mkdir data/elais
tar -zxvf elais_11JAN2014_2x2_5x5.tgz -C data/elais
mkdir data/cdfs
tar -zxvf cdfs_11JAN2014_2x2_5x5.tgz -C data/cdfs
cp RGZ-ATLAS-Bookkeeping.csv data/RGZ-ATLAS-Bookkeeping.csv
