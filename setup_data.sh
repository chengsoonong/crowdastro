mkdir data
tar -zxvf sanitized_radio_2016-03-01.tar.gz -C data
gunzip -c ATLASDR3_cmpcat_23July2015.dat.gz > data/ATLASDR3_cmpcat_23July2015.dat
mkdir data/elais
tar -zxvf elais_11JAN2014_2x2_5x5.tgz -C data/elais
mkdir data/cdfs
tar -zxvf cdfs_11JAN2014_2x2_5x5.tgz -C data/cdfs
