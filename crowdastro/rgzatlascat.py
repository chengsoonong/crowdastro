from astropy.io import ascii
import numpy as n
import math as m
import scipy
import csv

#----------------------------------------------------------------------------------
def ra2deg(hh,mm,ss):
  # converts RA from hh:mm:ss to decimal degrees

  # check values
  if hh > 24:
    print 'ERROR: RA must not exceed 24 hours\n'
    print 'exiting'

  radeg = (((float(ss)/60.) + float(mm))/60.) + float(hh)
  radeg = 15. * radeg

  return radeg

def dec2deg(dd,mm,ss):
  # converts DEC from dd:mm:ss to decimal degrees
  dd = float(dd)
  sign = "+"
  # check values
  if dd > 90:
    print 'ERROR: DEC must not exceed 90 degrees\n'
    print 'exiting'

  if dd < -90:
    print 'ERROR: DEC must not exceed -90 degrees\n'
    print 'exiting'

  # check sign of declination

  if dd < 0:
    sign = "-"
  
  decdeg = (((float(ss)/60.) + float(mm))/60.) + abs(dd)

  if sign == '-':
    decdeg = decdeg * -1

  return decdeg


#----------------------------------------------------------------------------------
# create output files
outhost = open('RGZ-ATLAS-CDFS-hosts.csv','w')

writer = csv.writer(outhost)
writer.writerow( ('zooniverse_id', 'rg_id','RA_H','RA_M','RA_S','DEC_D','DEC_M','DEC_S',\
  'RA_DEG','DEC_DEG','Agreement','S2','S2_err','Ncomp','SWIRE_ID','ra','dec','unc_ra','unc_dec',\
  'tile','flux_ap1_36','uncf_ap1_36','flux_ap2_36','uncf_ap2_36','flux_ap3_36','uncf_ap3_36',\
  'flux_ap4_36','uncf_ap4_36','flux_ap5_36','uncf_ap5_36','flux_kr_36','uncf_kr_36',\
  'rad_kr_36','flux_iso_36','uncf_iso_36','area_iso_36','stell_36','a_36','b_36','theta_36',\
  'flgs_36','ext_fl_36','cov_avg_36','cov_36','flux_ap1_45','uncf_ap1_45','flux_ap2_45',\
  'uncf_ap2_45','flux_ap3_45','uncf_ap3_45','flux_ap4_45','uncf_ap4_45','flux_ap5_45','uncf_ap5_45',\
  'flux_kr_45','uncf_kr_45','rad_kr_45','flux_iso_45','uncf_iso_45','area_iso_45','stell_45','a_45',\
  'b_45','theta_45','flgs_45','ext_fl_45','cov_avg_45','cov_45','flux_ap1_58','uncf_ap1_58',\
  'flux_ap2_58','uncf_ap2_58','flux_ap3_58','uncf_ap3_58','flux_ap4_58','uncf_ap4_58','flux_ap5_58',\
  'uncf_ap5_58','flux_kr_58',' uncf_kr_58','rad_kr_58','flux_iso_58','uncf_iso_58','area_iso_58',\
  'stell_58','a_58','b_58','theta_58','flgs_58','ext_fl_58','cov_avg_58','cov_58','flux_ap1_80',\
  'uncf_ap1_80','flux_ap2_80','uncf_ap2_80','flux_ap3_80','uncf_ap3_80','flux_ap4_80','uncf_ap4_80',\
  'flux_ap5_80','uncf_ap5_80','flux_kr_80','uncf_kr_80','rad_kr_80','flux_iso_80','uncf_iso_80',\
  'area_iso_80','stell_80','a_80','b_80','theta_80','flgs_80','ext_fl_80','cov_avg_80','cov_80',\
  'detid_24','flux_ap2_24','uncf_ap2_24','flux_ap3_24','uncf_ap3_24','flux_ap4_24','uncf_ap4_24',\
  'flux_ap5_24','uncf_ap5_24','flux_kr_24','uncf_kr_24','rad_kr_24','flux_iso_24','uncf_iso_24',\
  'area_iso_24','stell_24','a_24','b_24','theta_24','flgs_24','ext_fl_24','cov_avg_24','cov_24',\
  'fl_2mass','cov_u','cov_g','cov_r','cov_i','cov_z','ra_opt','dec_opt','ap_m_u','msig_u',\
  'ap_m_g','msig_g','ap_m_r','msig_r','ap_m_i','msig_i','ap_m_z','msig_z','int_m_u','int_m_g',\
  'int_m_r','int_m_i','int_m_z','fl_u','fl_g','fl_r','fl_i','fl_z','xid_p','xid_dist'))


outcomp = open('RGZ-ATLAS-CDFS-components.csv','w')
writer2 = csv.writer(outcomp)
writer2.writerow(('rgz_id','RA_H','RA_M','RA_S','DEC_D','DEC_M','DEC_S',\
  'RA_DEG','DEC_DEG','ATLAS_ID','ATLAS_Name','ATLAS_RA','ATLAS_DEC','ATLAS_RA_deg','ATLAS_DEC_deg',\
  'RA_err','DEC_err','rms','BWS','Obs_freq','Sp','Sp_err','S','S_err','Sp2','Sp2_err','S2','S2_err',\
  'Deconv','Deconv_err','class','Sindex','Sindex_err','field','Agreement'))

# SWIRE catalogue
# -- for some reason astropy can't read in the SWIRE ipac table
# -- added # to the ipac comments and column headers
swirefile = 'SWIRE3_CDFS_cat_IRAC24_21Dec05.tbl'
swiredata = ascii.read(swirefile)#,format='ipac',definition='ignore',guess=False)

swireid = n.array(swiredata['col2'])

# atlas RGZ files
hostfile = 'rgz_hosts_kde_24MAY2016.csv'
compfile = 'rgz_radio_components_kde_24MAY2106.csv'

# ATLAS catalogue from Franzen+(2015)
atlasfile = 'ATLASDR3_cmpcat_23July2015.csv'
atlasdata = ascii.read(atlasfile)

atlasid = atlasdata['ID']
atlasra = n.array(atlasdata['RA_deg'])
atlasdec = n.array(atlasdata['Dec_deg'])

# read in files
hostdata = ascii.read(hostfile)

zooid = hostdata['zooniverse_id']
source = hostdata['source']
rgzname = hostdata['rgz_name']
swirename = hostdata['swire_name']
ra = hostdata['ra']
dec = hostdata['dec']
agreement = hostdata['agreement']

compdata = ascii.read(compfile)
comprgzname = compdata['rgz_name']
compid = compdata['radio_component']
compagree = compdata['agreement']


# split RA and DEC values
rah = n.zeros(len(ra))
ram = n.zeros(len(ra))
ras = n.zeros(len(ra))
decd = n.zeros(len(dec))
decm = n.zeros(len(dec))
decs = n.zeros(len(dec))

ii = 0
while ii < len(ra):
  ## RA
  tmp = ra[ii].split("h")
  rah[ii] = int(tmp[0])

  tmp2 = tmp[1].split("m")
  ram[ii] = int(tmp2[0])

  tmp3 = tmp2[1].split("s")
  ras[ii] = float(tmp3[0])

  # convert to decimal degrees
  raindeg = ra2deg(rah[ii],ram[ii],ras[ii])

  tmp = ''
  tmp2 = ''
  tmp3 = ''
  
  ## DEC
  tmp = dec[ii].split("d")
  decd[ii] = int(tmp[0])

  tmp2 = tmp[1].split("m")
  decm[ii] = int(tmp2[0])

  tmp3 = tmp2[1].split("s")
  decs[ii] = float(tmp3[0])

  # convert to decimal degrees
  decindeg = dec2deg(decd[ii],decm[ii],decs[ii])
  tmp = ''
  tmp2 = ''
  tmp3 = ''

  # determine how many components there are for this RGZ subject
  indcomp = n.where(rgzname[ii] == comprgzname)
  ncomp = len(indcomp[0])

  # only one radio emitting structure for the host galaxy
  if ncomp == 1:
    # write the values out to the host catalogue --> this includes SWIRE values
    newind = n.where(compid[indcomp][0] == atlasdata['ID'])

    # locate SWIRE id in SWIRE catalogue
    swind = n.where(swirename[ii] == swireid)
    if (len(swind[0]) < 1) or (len(swind[0]) > 2):
      print 'SWIRE id invalid...exiting...'
      print ''
      exit()

    writer.writerow((zooid[ii], rgzname[ii], rah[ii],\
      ram[ii], ras[ii], decd[ii], decm[ii], decs[ii], raindeg, decindeg, agreement[ii], \
      atlasdata['S2'][newind][0], atlasdata['S2_err'][newind][0], ncomp, \
      swiredata['col2'][swind][0], swiredata['col3'][swind][0], \
      swiredata['col4'][swind][0], swiredata['col5'][swind][0], swiredata['col6'][swind][0], \
      swiredata['col7'][swind][0],  swiredata['col8'][swind][0], swiredata['col9'][swind][0],\
      swiredata['col10'][swind][0], swiredata['col11'][swind][0], swiredata['col12'][swind][0], \
      swiredata['col13'][swind][0], swiredata['col14'][swind][0], swiredata['col15'][swind][0], \
      swiredata['col16'][swind][0], swiredata['col17'][swind][0], swiredata['col18'][swind][0], \
      swiredata['col19'][swind][0], swiredata['col20'][swind][0], swiredata['col21'][swind][0], \
      swiredata['col22'][swind][0], swiredata['col23'][swind][0], swiredata['col24'][swind][0], \
      swiredata['col25'][swind][0], swiredata['col26'][swind][0], swiredata['col27'][swind][0], \
      swiredata['col28'][swind][0], swiredata['col29'][swind][0], swiredata['col30'][swind][0], \
      swiredata['col31'][swind][0], swiredata['col32'][swind][0], swiredata['col33'][swind][0], \
      swiredata['col34'][swind][0], swiredata['col35'][swind][0], swiredata['col36'][swind][0], \
      swiredata['col37'][swind][0], swiredata['col38'][swind][0], swiredata['col39'][swind][0], \
      swiredata['col40'][swind][0], swiredata['col41'][swind][0], swiredata['col42'][swind][0], \
      swiredata['col43'][swind][0], swiredata['col44'][swind][0], swiredata['col45'][swind][0], \
      swiredata['col46'][swind][0], swiredata['col47'][swind][0], swiredata['col48'][swind][0], \
      swiredata['col49'][swind][0], swiredata['col50'][swind][0], swiredata['col51'][swind][0], \
      swiredata['col52'][swind][0], swiredata['col53'][swind][0], swiredata['col54'][swind][0], \
      swiredata['col55'][swind][0], swiredata['col56'][swind][0], swiredata['col57'][swind][0], \
      swiredata['col58'][swind][0], swiredata['col59'][swind][0], swiredata['col60'][swind][0], \
      swiredata['col61'][swind][0], swiredata['col62'][swind][0], swiredata['col63'][swind][0], \
      swiredata['col64'][swind][0], swiredata['col65'][swind][0], swiredata['col66'][swind][0], \
      swiredata['col67'][swind][0], swiredata['col68'][swind][0], swiredata['col69'][swind][0], \
      swiredata['col70'][swind][0], swiredata['col71'][swind][0], swiredata['col72'][swind][0], \
      swiredata['col73'][swind][0], swiredata['col74'][swind][0], swiredata['col75'][swind][0], \
      swiredata['col76'][swind][0], swiredata['col77'][swind][0], swiredata['col78'][swind][0], \
      swiredata['col79'][swind][0], swiredata['col80'][swind][0], swiredata['col81'][swind][0], \
      swiredata['col82'][swind][0], swiredata['col83'][swind][0], swiredata['col84'][swind][0], \
      swiredata['col85'][swind][0], swiredata['col86'][swind][0], swiredata['col87'][swind][0], \
      swiredata['col88'][swind][0], swiredata['col89'][swind][0], swiredata['col90'][swind][0], \
      swiredata['col91'][swind][0], swiredata['col92'][swind][0], swiredata['col93'][swind][0], \
      swiredata['col94'][swind][0], swiredata['col95'][swind][0], swiredata['col96'][swind][0], \
      swiredata['col97'][swind][0], swiredata['col98'][swind][0], swiredata['col99'][swind][0], \
      swiredata['col100'][swind][0], swiredata['col101'][swind][0], swiredata['col102'][swind][0], \
      swiredata['col103'][swind][0], swiredata['col104'][swind][0], swiredata['col105'][swind][0], \
      swiredata['col106'][swind][0], swiredata['col107'][swind][0], swiredata['col108'][swind][0], \
      swiredata['col109'][swind][0], swiredata['col110'][swind][0], swiredata['col111'][swind][0], \
      swiredata['col112'][swind][0], swiredata['col113'][swind][0], swiredata['col114'][swind][0], \
      swiredata['col115'][swind][0], swiredata['col116'][swind][0], swiredata['col117'][swind][0], \
      swiredata['col118'][swind][0], swiredata['col119'][swind][0], swiredata['col120'][swind][0], \
      swiredata['col121'][swind][0], swiredata['col122'][swind][0], swiredata['col123'][swind][0], \
      swiredata['col124'][swind][0], swiredata['col125'][swind][0], swiredata['col126'][swind][0], \
      swiredata['col127'][swind][0], swiredata['col128'][swind][0], swiredata['col129'][swind][0], \
      swiredata['col130'][swind][0], swiredata['col131'][swind][0], swiredata['col132'][swind][0], \
      swiredata['col133'][swind][0], swiredata['col134'][swind][0], swiredata['col135'][swind][0], \
      swiredata['col136'][swind][0], swiredata['col137'][swind][0], swiredata['col138'][swind][0], \
      swiredata['col139'][swind][0], swiredata['col140'][swind][0], swiredata['col141'][swind][0], \
      swiredata['col142'][swind][0], swiredata['col143'][swind][0], swiredata['col144'][swind][0], \
      swiredata['col145'][swind][0], swiredata['col146'][swind][0], swiredata['col147'][swind][0], \
      swiredata['col148'][swind][0], swiredata['col149'][swind][0], swiredata['col150'][swind][0], \
      swiredata['col151'][swind][0], swiredata['col152'][swind][0], swiredata['col153'][swind][0], \
      swiredata['col154'][swind][0], swiredata['col155'][swind][0], swiredata['col156'][swind][0]))


    # write the values out to the component catalogue --> only ATLAS data
    writer2.writerow((rgzname[ii],rah[ii],ram[ii],ras[ii],decd[ii],decm[ii],decs[ii],raindeg,decindeg,atlasdata['ID'][newind][0],atlasdata['name'][newind][0],atlasdata['RA'][newind][0],atlasdata['Dec'][newind][0],atlasdata['RA_deg'][newind][0],atlasdata['Dec_deg'][newind][0],atlasdata['RA_err'][newind][0],atlasdata['Dec_err'][newind][0],atlasdata['rms'][newind][0],atlasdata['BWS'][newind][0],atlasdata['Obs_freq'][newind][0],atlasdata['Sp'][newind][0],atlasdata['Sp_err'][newind][0],atlasdata['S'][newind][0],atlasdata['S_err'][newind][0],atlasdata['Sp2'][newind][0],atlasdata['Sp2_err'][newind][0],atlasdata['S2'][newind][0],atlasdata['S2_err'][newind][0],atlasdata['Deconv'][newind][0],atlasdata['Deconv_err'][newind][0],atlasdata['class'][newind][0],atlasdata['Sindex'][newind][0],atlasdata['Sindex_err'][newind][0],atlasdata['field'][newind][0],compagree[ii]))


  # more then one radio emitting structure for the host galaxy
  if ncomp > 1:
    newflux = 0.0
    newfluxerr = 0.0
    for jj in n.arange(ncomp):
      newind = indcomp[0][jj]
      atlasind = n.where(compid[newind] == atlasdata['ID'])
    
      # determine the total flux for the radio emitting source
      newflux = atlasdata['S2'][atlasind][0] + newflux
      newfluxerr = (atlasdata['S2_err'][atlasind][0]*atlasdata['S2_err'][atlasind][0]) + newfluxerr

      ### need to calculate the error in the flux ###

      # write each radio component to the component catalogue --> only ATLAS data
      writer2.writerow((rgzname[ii],rah[ii],ram[ii],ras[ii],decd[ii],decm[ii],decs[ii],raindeg,decindeg,atlasdata['ID'][atlasind][0],atlasdata['name'][atlasind][0],atlasdata['RA'][atlasind][0],atlasdata['Dec'][atlasind][0],atlasdata['RA_deg'][atlasind][0],atlasdata['Dec_deg'][atlasind][0],atlasdata['RA_err'][atlasind][0],atlasdata['Dec_err'][atlasind][0],atlasdata['rms'][atlasind][0],atlasdata['BWS'][atlasind][0],atlasdata['Obs_freq'][atlasind][0],atlasdata['Sp'][atlasind][0],atlasdata['Sp_err'][atlasind][0],atlasdata['S'][atlasind][0],atlasdata['S_err'][atlasind][0],atlasdata['Sp2'][atlasind][0],atlasdata['Sp2_err'][atlasind][0],atlasdata['S2'][atlasind][0],atlasdata['S2_err'][atlasind][0],atlasdata['Deconv'][atlasind][0],atlasdata['Deconv_err'][atlasind][0],atlasdata['class'][atlasind][0],atlasdata['Sindex'][atlasind][0],atlasdata['Sindex_err'][atlasind][0],atlasdata['field'][atlasind][0],compagree[newind]))


    # write the combined radio components to the host file

    # locate SWIRE id in SWIRE catalogue
    swind = n.where(swirename[ii] == swireid)
    if (len(swind[0]) < 1) or (len(swind[0]) > 2):
      print 'SWIRE id invalid...exiting...'
      print ''
      exit()

    newfluxerr = n.sqrt(newfluxerr)

    writer.writerow((zooid[ii], rgzname[ii], rah[ii],\
      ram[ii], ras[ii], decd[ii], decm[ii], decs[ii], raindeg, decindeg, agreement[ii], \
      newflux, newfluxerr, ncomp, swiredata['col2'][swind][0], swiredata['col3'][swind][0], \
      swiredata['col4'][swind][0], swiredata['col5'][swind][0], swiredata['col6'][swind][0], \
      swiredata['col7'][swind][0],  swiredata['col8'][swind][0], swiredata['col9'][swind][0],\
      swiredata['col10'][swind][0], swiredata['col11'][swind][0], swiredata['col12'][swind][0], \
      swiredata['col13'][swind][0], swiredata['col14'][swind][0], swiredata['col15'][swind][0], \
      swiredata['col16'][swind][0], swiredata['col17'][swind][0], swiredata['col18'][swind][0], \
      swiredata['col19'][swind][0], swiredata['col20'][swind][0], swiredata['col21'][swind][0], \
      swiredata['col22'][swind][0], swiredata['col23'][swind][0], swiredata['col24'][swind][0], \
      swiredata['col25'][swind][0], swiredata['col26'][swind][0], swiredata['col27'][swind][0], \
      swiredata['col28'][swind][0], swiredata['col29'][swind][0], swiredata['col30'][swind][0], \
      swiredata['col31'][swind][0], swiredata['col32'][swind][0], swiredata['col33'][swind][0], \
      swiredata['col34'][swind][0], swiredata['col35'][swind][0], swiredata['col36'][swind][0], \
      swiredata['col37'][swind][0], swiredata['col38'][swind][0], swiredata['col39'][swind][0], \
      swiredata['col40'][swind][0], swiredata['col41'][swind][0], swiredata['col42'][swind][0], \
      swiredata['col43'][swind][0], swiredata['col44'][swind][0], swiredata['col45'][swind][0], \
      swiredata['col46'][swind][0], swiredata['col47'][swind][0], swiredata['col48'][swind][0], \
      swiredata['col49'][swind][0], swiredata['col50'][swind][0], swiredata['col51'][swind][0], \
      swiredata['col52'][swind][0], swiredata['col53'][swind][0], swiredata['col54'][swind][0], \
      swiredata['col55'][swind][0], swiredata['col56'][swind][0], swiredata['col57'][swind][0], \
      swiredata['col58'][swind][0], swiredata['col59'][swind][0], swiredata['col60'][swind][0], \
      swiredata['col61'][swind][0], swiredata['col62'][swind][0], swiredata['col63'][swind][0], \
      swiredata['col64'][swind][0], swiredata['col65'][swind][0], swiredata['col66'][swind][0], \
      swiredata['col67'][swind][0], swiredata['col68'][swind][0], swiredata['col69'][swind][0], \
      swiredata['col70'][swind][0], swiredata['col71'][swind][0], swiredata['col72'][swind][0], \
      swiredata['col73'][swind][0], swiredata['col74'][swind][0], swiredata['col75'][swind][0], \
      swiredata['col76'][swind][0], swiredata['col77'][swind][0], swiredata['col78'][swind][0], \
      swiredata['col79'][swind][0], swiredata['col80'][swind][0], swiredata['col81'][swind][0], \
      swiredata['col82'][swind][0], swiredata['col83'][swind][0], swiredata['col84'][swind][0], \
      swiredata['col85'][swind][0], swiredata['col86'][swind][0], swiredata['col87'][swind][0], \
      swiredata['col88'][swind][0], swiredata['col89'][swind][0], swiredata['col90'][swind][0], \
      swiredata['col91'][swind][0], swiredata['col92'][swind][0], swiredata['col93'][swind][0], \
      swiredata['col94'][swind][0], swiredata['col95'][swind][0], swiredata['col96'][swind][0], \
      swiredata['col97'][swind][0], swiredata['col98'][swind][0], swiredata['col99'][swind][0], \
      swiredata['col100'][swind][0], swiredata['col101'][swind][0], swiredata['col102'][swind][0], \
      swiredata['col103'][swind][0], swiredata['col104'][swind][0], swiredata['col105'][swind][0], \
      swiredata['col106'][swind][0], swiredata['col107'][swind][0], swiredata['col108'][swind][0], \
      swiredata['col109'][swind][0], swiredata['col110'][swind][0], swiredata['col111'][swind][0], \
      swiredata['col112'][swind][0], swiredata['col113'][swind][0], swiredata['col114'][swind][0], \
      swiredata['col115'][swind][0], swiredata['col116'][swind][0], swiredata['col117'][swind][0], \
      swiredata['col118'][swind][0], swiredata['col119'][swind][0], swiredata['col120'][swind][0], \
      swiredata['col121'][swind][0], swiredata['col122'][swind][0], swiredata['col123'][swind][0], \
      swiredata['col124'][swind][0], swiredata['col125'][swind][0], swiredata['col126'][swind][0], \
      swiredata['col127'][swind][0], swiredata['col128'][swind][0], swiredata['col129'][swind][0], \
      swiredata['col130'][swind][0], swiredata['col131'][swind][0], swiredata['col132'][swind][0], \
      swiredata['col133'][swind][0], swiredata['col134'][swind][0], swiredata['col135'][swind][0], \
      swiredata['col136'][swind][0], swiredata['col137'][swind][0], swiredata['col138'][swind][0], \
      swiredata['col139'][swind][0], swiredata['col140'][swind][0], swiredata['col141'][swind][0], \
      swiredata['col142'][swind][0], swiredata['col143'][swind][0], swiredata['col144'][swind][0], \
      swiredata['col145'][swind][0], swiredata['col146'][swind][0], swiredata['col147'][swind][0], \
      swiredata['col148'][swind][0], swiredata['col149'][swind][0], swiredata['col150'][swind][0], \
      swiredata['col151'][swind][0], swiredata['col152'][swind][0], swiredata['col153'][swind][0], \
      swiredata['col154'][swind][0], swiredata['col155'][swind][0], swiredata['col156'][swind][0]))


  # no radio components???
  if ncomp < 1:
    print 'no components for this RGZ subject????'
    print ''
    print rgzname[ii],agreement[ii],ncomp
    print ''
    print 'exiting'
    exit()

 
  ii = ii + 1


outhost.close()
outcomp.close()

