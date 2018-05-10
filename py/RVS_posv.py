
#
# RVS_posv
#
# reading DR/RVS*.fits
#

import pyfits
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from matplotlib import patches
from scipy import stats
from scipy import optimize
import scipy.interpolate
from galpy.util import bovy_coords, bovy_plot

# for not displaying
matplotlib.use('Agg')

##### main programme start here #####

# constant for proper motion unit conversion
pmvconst = 4.74047
# from Bland-Hawthorn & Gerhard
usun = 10.0
vsun = 248.0
wsun = 7.0
vcircsun = 248.0-11.0
rsun = 8.2
zsun = 0.025

# condition to select stars 
e_plxlim = 0.15
zmaxlim = 0.2
glonlow = 0.0
glonhigh = 360.0
vloserrlim = 1000.0
# minimum plx
plxlim=0.001

# read data
ras = np.array([])
decs = np.array([])
glons = np.array([])
glats = np.array([])
plxs_obs = np.array([])
pmras_obs = np.array([])
pmdecs_obs = np.array([])
# HRV
hrvs_obs = np.array([])
# G, G_BP, G_RP
gmag_obs = np.array([])
gbpmag_obs = np.array([])
grpmag_obs = np.array([])

nfile = 8
for ii in range(nfile):
    # RVS data
    if ii == 0:
        infilel = 'DR2/RVSl0-45-result.fits'
    elif ii == 1:
        infilel = 'DR2/RVSl45-90-result.fits'
    elif ii == 2:
        infilel = 'DR2/RVSl90-135-result.fits'
    elif ii == 3:
        infilel = 'DR2/RVSl135-180-result.fits'
    elif ii == 4:
        infilel = 'DR2/RVSl180-225-result.fits'
    elif ii == 5:
        infilel = 'DR2/RVSl225-270-result.fits'
    elif ii == 6:
        infilel = 'DR2/RVSl270-315-result.fits'
    else:
        infilel = 'DR2/RVSl315-360-result.fits'

    star_hdus = pyfits.open(infilel)
    star = star_hdus[1].data
    star_hdus.close()

    gabsmag = star['phot_g_mean_mag'] \
        -(5.0*np.log10(100.0/np.fabs(star['parallax']))) \
        +star['a_g_val']
    zabs = np.fabs((1.0/star['parallax']) \
        *np.sin(np.pi*star['b']/180.0)+zsun)
    yabs = np.fabs((1.0/star['parallax']) \
        *np.sin(np.pi*star['l']/180.0))

    # minimum distance limit
    distmin = 0.0000000001

    sindx = np.where((zabs < zmaxlim) & 
        (star['parallax']>0.0) & (star['parallax']<1.0/distmin) & 
        (star['parallax_error']/star['parallax']<e_plxlim) & 
        (star['radial_velocity_error']>0.0) &
        (star['radial_velocity_error']<vloserrlim) &
        (np.logical_and(star['l']>glonlow,star['l']<glonhigh)))

    print 'Reading',infilel,' N Selected=',len(star['ra'][sindx])
    # store the stellar data
    ras = np.append(ras, star['ra'][sindx])
    decs = np.append(decs, star['dec'][sindx])
    glons = np.append(glons, star['l'][sindx])
    glats = np.append(glats, star['b'][sindx])
    plxs_obs = np.append(plxs_obs, star['parallax'][sindx])
    pmras_obs = np.append(pmras_obs, star['pmra'][sindx])
    pmdecs_obs = np.append(pmdecs_obs, star['pmdec'][sindx])
    # HRV
    hrvs_obs = np.append(hrvs_obs, star['radial_velocity'][sindx])
    # G, G_BP, G_RP
    gmag_obs = np.append(gmag_obs, star['phot_g_mean_mag'][sindx])
    gbpmag_obs = np.append(gbpmag_obs, star['phot_bp_mean_mag'][sindx])
    grpmag_obs = np.append(grpmag_obs, star['phot_rp_mean_mag'][sindx])

print ' Total number of selected stars = ',len(gmag_obs)

# convert deg -> rad
glonrads = glons*np.pi/180.0
glatrads = glats*np.pi/180.0

# get observed position and velocity
dists_obs = 1.0/plxs_obs

# velocity
Tpmllpmbb = bovy_coords.pmrapmdec_to_pmllpmbb( \
    pmras_obs, pmdecs_obs, ras, \
    decs, degree=True, epoch=None)
pmlons_obs = Tpmllpmbb[:,0]
pmlats_obs = Tpmllpmbb[:,1]
# mas/yr -> km/s
vlons_obs = pmvconst*pmlons_obs*dists_obs
vlats_obs = pmvconst*pmlats_obs*dists_obs
# galactic position
distxys_obs = dists_obs*np.cos(glatrads)
xpos_obs = distxys_obs*np.cos(glonrads)
ypos_obs = distxys_obs*np.sin(glonrads)
zpos_obs = dists_obs*np.sin(glatrads)
xposgals_obs = xpos_obs-rsun
yposgals_obs = ypos_obs
zposgals_obs = zpos_obs+zsun
rgals_obs = np.sqrt(xposgals_obs**2+yposgals_obs**2)

# to vx vy vz
Tvxvyvz = bovy_coords.vrpmllpmbb_to_vxvyvz(\
    hrvs_obs, Tpmllpmbb[:,0], Tpmllpmbb[:,1], \
    glons, glats, dists_obs, XYZ=False, degree=True)
vxs_obs = Tvxvyvz[:,0]
vys_obs = Tvxvyvz[:,1]
vzs_obs = Tvxvyvz[:,2]+wsun
# Galactocentric position and velcoity
hrvxys_obs = hrvs_obs*np.cos(glatrads)
vxgals_obs = vxs_obs+usun
vygals_obs = vys_obs+vsun
vrots_obs = (vxgals_obs*yposgals_obs-vygals_obs*xposgals_obs) \
    /rgals_obs
vrads_obs = (vxgals_obs*xposgals_obs+vygals_obs*yposgals_obs) \
    /rgals_obs

# fits output
tbhdu = pyfits.BinTableHDU.from_columns([\
  pyfits.Column(name='X',unit='(kpc)',format='D',array=xposgals_obs), \
  pyfits.Column(name='Y',unit='(kpc)',format='D',array=yposgals_obs), \
  pyfits.Column(name='Z',unit='(kpc)',format='D',array=zposgals_obs), \
  pyfits.Column(name='Vx',unit='(km/s)',format='D',array=vxgals_obs), \
  pyfits.Column(name='Vy',unit='(km/s)',format='D',array=vygals_obs), \
  pyfits.Column(name='Vz',unit='(km/s)',format='D',array=vzs_obs), \
  pyfits.Column(name='Vrot',unit='(km/s)',format='D',array=vrots_obs), \
  pyfits.Column(name='Vrad',unit='(km/s)',format='D',array=vrads_obs)])
tbhdu.writeto('RVS_posv.fits',clobber=True)

