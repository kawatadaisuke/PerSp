
# 1. Read /Users/dkawata/work/obs/LAMOST/DR3/dr3_stellar.fits
# 2. output sels_rv.asc for gcdp-ana/lbsels.dat for mock data generation
# 3. Plot x-y distribution 
#
# History:
#  29/03/2018  Written - Daisuke Kawata
#

import pyfits
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import mwdust
from scipy import stats
from galpy.util import bovy_coords

# teff vs. Mv
# F0V-A5V
#       F0V     A9V
teffmv=np.array([7220.0, 7440.0, 7500.0, 7800.0, 8000.0, 8080.0])
mvmag= np.array([2.51,   2.30,   2.29,   2.07,   1.89,   1.84])
bvcol= np.array([0.294,  0.255,  0.250,  0.210,  0.170,  0.160])
# print ' mvmag and bvcol shape=',np.shape(mvmag),np.shape(bvcol)
print ' Mv =',mvmag
# Jester et al. (2005) http://www.sdss3.org/dr8/algorithms/sdssUBVRITransform.php
# not use SDSS photometry

# input data
infile='/Users/dkawata/work/obs/LAMOST/DR3/LAMOSTDR3_AstarxAPASSDR9.fits'
star_hdus=pyfits.open(infile)
star=star_hdus[1].data
star_hdus.close()

# read the data
# number of data points
print 'number of stars read =',len(star['obsid'])

# select stas with teff and logg
# Galactic coordinates
Tllbb=bovy_coords.radec_to_lb(star['ra'],star['dec'],degree=True,epoch=2000.0)
glon=Tllbb[:,0]
glat=Tllbb[:,1]

sindx=np.where((star['teff']>7330.0) & (star['teff']<8040.0) \
               & (star['logg']>3.2) \
               & (star['Vmag']>0.0) & (star['Bmag']>0.0) \
               & (star['rv_err']>0.0) & (star['rv_err']<10.0) \
               & (glon>140.0) & (glon<220.0))
#               & (glon>175.0) & (glon<185.0))

nstars=len(star['ra'][sindx])
print ' N selected=',nstars
# extract the necessary particle info
ra_s=star['ra'][sindx]
dec_s=star['dec'][sindx]
teff_s=star['teff'][sindx]
logg_s=star['logg'][sindx]
# from APASS DR9
vmag_s=star['Vmag'][sindx]
bmag_s=star['Bmag'][sindx]
feh_s=star['feh'][sindx]
rv_s=star['rv'][sindx]
rverr_s=star['rv_err'][sindx]
glon_s=glon[sindx]
glat_s=glat[sindx]

# absolute R mag
mvmag_s=np.interp(teff_s,teffmv,mvmag)
# extinction
combined=mwdust.Combined15(filter='CTIO V')
avmag=np.zeros_like(glon_s)
mod0_s=vmag_s-mvmag_s+avmag
dist0_s=np.power(10.0,(mod0_s+5.0)/5.0)*0.001
dist_s=np.power(10.0,(mod0_s+5.0)/5.0)*0.001
for i in range(0):
  # distance modulus
  mod_s=vmag_s-mvmag_s-avmag
  dist_s=np.power(10.0,(mod_s+5.0)/5.0)*0.001
  # calculate extinction
  for j in range(len(glon_s)):
    avmag[j]=combined(glon_s[j],glat_s[j],dist_s[j])
  print ' mwdust iteration ',i,' finished'

# photometry V and V-I
# dwarf http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt
# follows V-I=B-V well up to B-V<1.5. Hence, in this purpose set V-I=B-V
vicol_s=bmag_s-vmag_s

# labes
# plt.xlabel(r"Teff",fontsize=18,fontname="serif")
# plt.ylabel(r"Mv (mag)",fontsize=18,fontname="serif",style="normal")
# scatter plot
# plt.scatter(teff_s,mvmag_s,c=dist_s,s=30,vmin=0.0,vmax=10.0)
# plt.show()

# Sun's radius used in Bland-Hawthorn & Gerhard (2016)
xsun=-8.1
# Sun's proper motion Schoenrich et al.
usun=11.1
vsun=12.24
wsun=7.25
# circular velocity
# Jo Bovy's suggestion
vcirc=30.24*np.abs(xsun)-vsun

# degree to radian
glonrad_s=glon_s*np.pi/180.0
glatrad_s=glat_s*np.pi/180.0
# x,y position
xpos_s=xsun+np.cos(glonrad_s)*dist_s*np.cos(glatrad_s)
ypos_s=np.sin(glonrad_s)*dist_s*np.cos(glatrad_s)
zpos_s=np.sin(glatrad_s)*dist_s
# rgal with Reid et al. value
rgal_s=np.sqrt(xpos_s**2+ypos_s**2)

# linear regression of metallicity gradient
slope, intercept, r_value, p_value, std_err = stats.linregress(rgal_s,feh_s)
print ' slope, intercept=',slope,intercept

# delta feh
delfeh_s=feh_s-(slope*rgal_s+intercept)

# output ascii data for test
f=open('star_pos.asc','w')
for i in range(nstars):
  print >>f, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f" \
    %(xpos_s[i],ypos_s[i],zpos_s[i],rgal_s[i] \
      ,feh_s[i],delfeh_s[i],glon_s[i],glat_s[i],dist_s[i],dist0_s[i] \
      ,avmag[i],bmag_s[i],vmag_s[i],vicol_s[i])
f.close()

# selecting the stars with z and Glon
# 3.75 kpc  15% plx accuracy with 0.04 mas plx error. 
distlim=3.75
sindxz=np.where((np.fabs(zpos_s)<0.5) & (dist_s<distlim))
nsels=len(rgal_s[sindxz])
print 'N stars(|z|<0.5 & d<',distlim,' kpc)=',nsels

# use position from the Sun
xsels=xpos_s[sindxz]-xsun
ysels=ypos_s[sindxz]
zsels=zpos_s[sindxz]
rvsels=rv_s[sindxz]
rverrsels=rverr_s[sindxz]
vmagsels=vmag_s[sindxz]
vicolsels=vicol_s[sindxz]
rasels=ra_s[sindxz]
decsels=dec_s[sindxz]
glonsels=glon_s[sindxz]
glatsels=glat_s[sindxz]
distsels=dist_s[sindxz]
teffsels=teff_s[sindxz]
loggsels=logg_s[sindxz]
avmagsels=avmag[sindxz]

# for input of lbsels
f=open('sels_rv.asc','w')
print >>f,"# nstar= %10d" % (nsels)
for i in range(nsels):
  print >>f,"%12.5e %12.5e %12.5e %12.5e %12.5e %12.5e %12.5e %12.5e %12.5e %12.5e %12.5e %12.5e %12.5e %12.5e %12.5e" \
    %(xsels[i],ysels[i],zsels[i],rvsels[i],rverrsels[i] \
      ,vmagsels[i],vicolsels[i],rasels[i],decsels[i],glonsels[i] \
      ,glatsels[i],distsels[i],teffsels[i],loggsels[i],avmagsels[i])
f.close()

### plot radial metallicity distribution
# plot Cepheids data point
plt.scatter(rgal_s[sindxz],feh_s[sindxz],c=delfeh_s[sindxz],s=5,vmin=-0.1,vmax=0.25,cmap=cm.jet)
# radial gradient
nsp=10
xsp=np.linspace(4.0,20.0,nsp)
ysp=slope*xsp+intercept
plt.plot(xsp,ysp,'b-')
plt.xlabel(r"R (kpc)",fontsize=18,fontname="serif")
plt.ylabel(r"[Fe/H]",fontsize=18,fontname="serif")
plt.axis([4.0,20.0,-1.0,0.75],'scaled')
cbar=plt.colorbar()
cbar.set_label(r'$\delta$[Fe/H]')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

# plot circle,
an=np.linspace(0,2.0*np.pi,100)
rad=7.0
i=0
rad=4.0
while i<15:
  rad=rad+0.5
  plt.plot(rad*np.cos(an),rad*np.sin(an),'k:')
  i+=1
# plot arm position from Reid et al. 2014
# number of points
nsp=100
isp=0
numsp=3
while isp<numsp:
# angle in R14 is clock-wise start at the Sun at (0.0, Rsun)
# convert to the one anti-clockwise starting from +x, y=0
  if isp==0:
# Scutum Arm
    angen=(180.0-3.0)*np.pi/180.0
#    angen=(180.0+45.0)*np.pi/180.0
    angst=(180.0-101.0)*np.pi/180.0
    angref=(180.0-27.6)*np.pi/180.0
    rref=5.0
# pitchangle
    tanpa=np.tan(19.8*np.pi/180.0)
  elif isp==1:
# Sagittarius Arm
    angen=(180.0+2.0)*np.pi/180.0
#    angen=(180.0+45.0)*np.pi/180.0
    angst=(180.0-68.0)*np.pi/180.0
    angref=(180.0-25.6)*np.pi/180.0
    rref=6.6
# pitchangle
    tanpa=np.tan(6.9*np.pi/180.0)
  else:
# Perseus Arm
    angen=(180.0-88.0)*np.pi/180.0
    angst=(180.0+21.0)*np.pi/180.0
    angref=(180.0-14.2)*np.pi/180.0
    rref=9.9
# pitchangle
    tanpa=np.tan(9.4*np.pi/180.0)
# logarithmic spiral arm , log r= tan(pa) theta, in the case of anti-clockwise arm
  an=np.linspace(angst,angen,nsp)
  xsp=np.zeros(nsp)
  ysp=np.zeros(nsp)
  i=0
  while i<nsp:
    rsp=np.exp(tanpa*(an[i]-angref))*rref
    xsp[i]=rsp*np.cos(an[i])
    ysp[i]=rsp*np.sin(an[i])
    i+=1
  if isp==0:
    plt.plot(xsp,ysp,'b-')
  elif isp==1:
    plt.plot(xsp,ysp,'r-')
  else:
    plt.plot(xsp,ysp,'g-')
  isp+=1

# plot Cepheids data point
plt.scatter(xsun,0.0,marker="*",s=100,color='k')
plt.scatter(xpos_s[sindxz],ypos_s[sindxz],c=delfeh_s[sindxz],s=10,vmin=-0.5,vmax=0.5,cmap=cm.jet)
plt.xlabel(r"X (kpc)",fontsize=18,fontname="serif")
plt.ylabel(r"Y (kpc)",fontsize=18,fontname="serif")
plt.axis([-13.0,-3.0,-4.5,4.5],'scaled')
cbar=plt.colorbar()
cbar.set_label(r'$\delta$[Fe/H]')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()
