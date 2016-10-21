from astropy.coordinates import SkyCoord
import numpy
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1, projection='mollweide')
ax.set_xticklabels(['14h','16h','18h','20h','22h','0h','2h','4h','6h','8h','10h'])
ax.grid(True, which='major')

# # # EMU
# # emu = SkyCoord(ra='-180', dec='-90', unit='deg')
# # ax.add_patch(
# #     patches.Rectangle(
# #         (-numpy.pi, emu.dec.rad),
# #         numpy.pi * 2,
# #         120 / 180 * numpy.pi,
# #         color='lightgreen',
# #     )
# # )


# # SWIRE
# centres = [
#     # ELAIS-S1
#     SkyCoord(ra='00 38 30', dec='-44 00 00', unit=('hour', 'deg')),
#     # XMM-LSS
#     SkyCoord(ra='02 21 00', dec='-04 30 00', unit=('hour', 'deg')),
#     # CDFS
#     SkyCoord(ra='03 32 00', dec='-28 16 00', unit=('hour', 'deg')),
#     # Lockman
#     SkyCoord(ra='10 45 00', dec='+58 00 00', unit=('hour', 'deg')),
#     # Lonsdale
#     SkyCoord(ra='14 41 00', dec='+59 25 00', unit=('hour', 'deg')),
#     # ELAIS-N1
#     SkyCoord(ra='16 11 00', dec='+55 00 00', unit=('hour', 'deg')),
#     # ELAIS-N2
#     SkyCoord(ra='16 36 48', dec='+41 01 45', unit=('hour', 'deg')),
# ]
# areas = [
#     # ELAIS-S1
#     14.26 * (numpy.pi / 180) ** 2,
#     # XMM-LSS
#     8.70 * (numpy.pi / 180) ** 2,
#     # CDFS
#     6.58 * (numpy.pi / 180) ** 2,
#     # Lockman
#     14.26 * (numpy.pi / 180) ** 2,
#     # Lonsdale
#     6.69 * (numpy.pi / 180) ** 2,
#     # ELAIS-N1
#     8.7 * (numpy.pi / 180) ** 2,
#     # ELAIS-N2
#     4.01 * (numpy.pi / 180) ** 2,
# ]
# names = ['ELAIS-S1', 'XMM-LSS', 'CDFS', 'Lockman', 'Lonsdale', 'ELAIS-N1',
#          'ELAIS-N2']
# font = {'family' : 'serif',
#         'size'   : 10}
# matplotlib.rc('font', **font)
# for centre, area, name in zip(centres, areas, names):
#     x = centre.ra.rad - numpy.sqrt(area) / 2
#     y = centre.dec.rad - numpy.sqrt(area) / 2

#     if x > numpy.pi:
#         x -= 2 * numpy.pi

#     ax.add_patch(
#         patches.Rectangle(
#             (x, y),
#             numpy.sqrt(area),
#             numpy.sqrt(area),
#             color='red',
#         )
#     )
#     ax.annotate(
#         name, xy=(x + numpy.sqrt(area), y + numpy.sqrt(area)),
#         xytext=(x + numpy.sqrt(area), y + numpy.sqrt(area)),
#     )


# FIRST
first = SkyCoord(ra='07.7', dec='28', unit=('hour', 'deg'))
# 01 59 57.183 -11 26 37.15 0.014

first_path = '../../data/first_catalog_14dec17.bin'
ras = []
decs = []
with open(first_path) as f:
    f.readline()
    f.readline()
    for line in f:
        if numpy.random.random() < 0.90:
            continue
        cols = line.split()
        rh, rm, rs, dd, dm, ds = cols[:6]
        coord = SkyCoord(
            ra='{}h{}m{}s'.format(rh,rm,rs),
            dec='{}d{}m{}s'.format(dd,dm,ds))
        ras.append(coord.ra.rad - numpy.pi)
        decs.append(coord.dec.rad)
plt.scatter(ras, decs, marker='.', color='blue', alpha=0.5, s=10)

# legend
emu_patch = patches.Patch(color='lightgreen', label='EMU')
first_patch = patches.Patch(color='lightblue', label='FIRST')
swire_patch = patches.Patch(color='red', label='SWIRE')
wise_patch = patches.Patch(fill=False, hatch='//', alpha=0.2, label='WISE')
ax.legend(handles=[emu_patch, first_patch, swire_patch, wise_patch], bbox_to_anchor=(0, 1.2, 1.2, 0), loc=3, ncol=2, borderaxespad=0.)

plt.show()
