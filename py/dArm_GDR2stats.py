
#
# darm_GDRstats.py
#
#  read gaia DR2 data and analyse velocity properties around the Perseus Arm
#

import pyfits
import math
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from scipy import stats
from galpy.util import bovy_coords
from galpy.util import bovy_plot
from mpi4py import MPI

comm=MPI.COMM_WORLD
nprocs=comm.Get_size()
myrank=comm.Get_rank()

# for not displaying
matplotlib.use('Agg')

# MPI communication process
def gather_allindx(ntotal, vals_rank, indx):
    # gather 
    vals = np.zeros(ntotal)
    ncom = ntotal
    vals[indx] = vals_rank
    sendbuf = np.zeros(ncom,dtype=np.float64)
    sendbuf = vals
    recvbuf = np.zeros(ncom,dtype=np.float64)
    comm.Allreduce(sendbuf,recvbuf,op=MPI.SUM)

    return recvbuf

# define position of arm

def funcdarm2(x, armp, xposp, yposp):
    # Arm parameters
    angref, rref, tanpa = armp
# Galactic parameters
    rsp = np.exp(tanpa * (x - angref)) * rref

    darm2 = (rsp * np.cos(x) - xposp)**2 + (rsp * np.sin(x) - yposp)**2

#  print 'darm2,rsp,ang,xposp,yposp=',np.sqrt(darm2),rsp,x,xposp,yposp

    return darm2

# define computing the distance to the arm


def distarm(tanpa, angref, rref, xpos, ypos):

    # find minimum
    darm = np.zeros(len(xpos))
    angmin = np.zeros(len(xpos))

    for i in range(len(xpos)):
        armp = angref[i], rref[i], tanpa[i]
        res = minimize_scalar(funcdarm2, args=(armp, xpos[i], ypos[i]), bounds=(
            0.0, 1.5 * np.pi), method='bounded')
        if res.success == False:
            if myrank==0:
                print ' no minimize result at i,x,y=', i, xpos[i], ypos[i]
                print ' res=', res
        darm[i] = np.sqrt(res.fun)
        angmin[i] = res.x

    return darm, angmin

# compute the angular distance from the arm at a fixed radius

def dangarm(tanpa, angref, rref, xpos, ypos):
    # find radius and angle (0 at x>0 and anti-clockwise)
    rgalpos=np.sqrt(xpos**2+ypos**2)
    angpos=np.arccos(xpos/rgalpos)
    angpos[ypos<0.0]=2.0*np.pi-angpos[ypos<0.0]
    # find angle
    angsp=angref+(np.log(rgalpos)-np.log(rref))/tanpa
    angdiff=angpos-angsp
    dangdiff=rgalpos*angdiff

    # test output
    # f = open('dangarm.asc', 'w')
    # print >>f, "# x y darm darm_sig Umean Usig Vmean Vsig"
    # for i in range(len(rgalpos)):
    #    print >>f, "%f %f %f %f %f %f %f %f %f" % (
    #        xpos[i], ypos[i], rgalpos[i], angpos[i], angsp[i], angdiff[i], \
    #        dangdiff[i],rgalpos[i]*np.cos(angsp[i]),rgalpos[i]*np.sin(angsp[i]))
    # f.close()

    return dangdiff,angdiff

##### starting main programme #####

# options
MCsample_v = False
MCsample_vgalp = False
# Selected glon range
# 0:l=105-135, 1:l=135-165, 2:l=165-195, 3:l=195-225
SGlon = 0

if SGlon == 0:
    glonlab = 'l105-135'
elif SGlon == 1:
    glonlab = 'l135-165'
elif SGlon == 2:
    glonlab = 'l165-195'
else:
    glonlab = 'l195-225'

if MCsample_vgalp == True:
    MCsample_v = True

if myrank == 0:
    print ' MCsample_v, MCsample_vgalp=', MCsample_v, MCsample_vgalp

# Reid et al. (2014)'s Galactic parameters
# Sun's position
rsunr14 = 8.34

if myrank == 0:
    print ' Rsun in Reid et al. (2014)=', rsunr14

# Glon range
if SGlon == 0:
    if myrank == 0:
        print ' Selected l range = 105 - 135'
    glonlow = 105
    glonhigh = 135
elif SGlon == 1:
    if myrank == 0:
        print ' Selected l range = 135 - 165'
    glonlow = 135
    glonhigh = 165
elif SGlon == 2:
    if myrank == 0:
        print ' Selected l range = 165 - 195'
    glonlow = 165
    glonhigh = 195
else:
    if myrank == 0:
        print ' Selected l range = 195 - 225'
    glonlow = 195
    glonhigh = 225

# Galactic parameters and uncertainties
# Bland-Hawthorn & Gerhard (2016)
rsun = 8.2
rsunsig = 0.1
# vertical position of the Sun
zsun = 0.025
zsunsig = 0.05
# angular speed of the Sun
omgsun = 30.24
omgsunsig = 0.12
# Sun's proper motion Schoenrich et al.
usun = 10.0
usunsig = 1.0
vsun = 11.0
vsunsig = 2.0
wsun = 7.0
wsunsig = 0.5
# Feast & Whitelock (1997)
# dvcdr = -2.4
# dvcdrsig = 1.2
dvcdr = 0.0
dvcdrsig = 1.2
# Vcirc
vcirc = omgsun * rsun
pmvconst = 4.74047


if myrank == 0:
    print '### Assumed Galactic parameters'
    print ' Rsun=', rsun, '+-', rsunsig, ' kpc'
    print ' zsun=', zsun, '+-', zsunsig, ' kpc'
    print ' Omega_sun=', omgsun, '+-', omgsunsig, ' km/s/kpc'
    print ' Solar proper motion U =', usun, '+-', usunsig, ' km/s'
    print ' Solar proper motion V =', vsun, '+-', vsunsig, ' km/s'
    print ' Solar proper motion W =', wsun, '+-', wsunsig, ' km/s'
    print ' dVc/dR =', dvcdr, '+-', dvcdrsig, ' km/s/kpc'
    print ' default Vcirc=', vcirc, ' km/s'

# if MCsample_vgalp == False:
    # if myrank == 0:
    #     print ' Because the arm position will not be adjusted, rsun set to be Rr14=', rsunr14
    # rsun=rsunr14

# read RVS data
if SGlon == 0:
    # Selected l range = 105 - 135
    infilel0 = 'DR2/RVSl90-135-result.fits'
    infilel1 = 'DR2/RVSl135-180-result.fits'
elif SGlon == 1:
    # Selected l range = 135 - 165
    infilel0 = 'DR2/RVSl90-135-result.fits'
    infilel1 = 'DR2/RVSl135-180-result.fits'
elif SGlon == 2:
    # Selected l range = 165 - 195
    infilel0 = 'DR2/RVSl135-180-result.fits'
    infilel1 = 'DR2/RVSl180-225-result.fits'
else:
    # Selected l range = 195 - 225
    infilel0 = 'DR2/RVSl180-225-result.fits'
    infilel1 = 'DR2/RVSl225-270-result.fits'
starl0 = pyfits.open(infilel0)
starl1 = pyfits.open(infilel1)
nrowsl0 = starl0[1].data.shape[0]
nrowsl1 = starl1[1].data.shape[0]
nrows = nrowsl0 + nrowsl1
star_hdu = pyfits.BinTableHDU.from_columns(starl0[1].columns, nrows=nrows)
for colname in starl0[1].columns.names:
    star_hdu.data[colname][nrowsl0:] = starl1[1].data[colname]
star = star_hdu.data
starl0.close()
starl1.close()

# select stars with e_Plx/Plx_obs<0.15
# select only velocity error is small enough
Verrlim = 10000.0
zmaxlim = 0.2
fPlxlim = 0.15
distmaxlim = 10.0
Plx_obs=star['parallax']
e_Plx=star['parallax_error']
pmRA_obs=star['pmra']
e_pmRA=star['pmra_error']
pmDEC_obs=star['pmdec']
e_pmDEC=star['pmdec_error']

zwerr = (1.0 / (Plx_obs-e_Plx) ) * np.sin(star['b'] * np.pi / 180.0) + zsun
# from Hunt et al. (2014)
errVra = pmvconst * np.sqrt( (e_pmRA**2+( (pmRA_obs*e_pmRA/Plx_obs)**2 ) ) \
  / Plx_obs**2)
errVdec = pmvconst*np.sqrt((e_pmDEC**2+((pmDEC_obs*e_pmDEC/Plx_obs)**2)) \
  /Plx_obs**2)
verr = np.sqrt(errVra**2 + errVdec**2 + star['radial_velocity_error']**2)
dist = 1.0/Plx_obs
sindx = np.where( (Plx_obs>0.0) & (e_Plx/Plx_obs<fPlxlim) &
   (np.abs(zwerr) < zmaxlim) & (dist < distmaxlim) &
   (star['l']>=glonlow) & (star['l']<glonhigh))

# extract the necessary stellar info
glonv = star['l'][sindx]
glatv = star['b'][sindx]
# number of data points
nstarv = len(glatv)
if myrank == 0:
    print ' number of stars selected=', nstarv
plxv = star['parallax'][sindx]
errplxv = star['parallax_error'][sindx]
distv = 1.0/plxv
# RA, DEC from Gaia data
rav = star['ra'][sindx]
decv = star['dec'][sindx]
pmrav = star['pmra'][sindx]
pmdecv = star['pmdec'][sindx]
errpmrav = star['pmra_error'][sindx]
errpmdecv = star['pmdec_error'][sindx]
plxpmra_corrv = star['parallax_pmra_corr'][sindx]
plxpmdec_corrv = star['parallax_pmdec_corr'][sindx]
pmradec_corrv = star['pmra_pmdec_corr'][sindx]

# error correlation check
# plxpmra_corrv = np.ones_like(pmradec_corrv)*0.5
# pmradec_corrv = np.ones_like(pmradec_corrv)*0.5

hrvv = star['radial_velocity'][sindx]
errhrvv = star['radial_velocity_error'][sindx]
# radian glon and glat
glonradv = glonv * np.pi / 180.0
glatradv = glatv * np.pi / 180.0

# x, y position
xposv = -rsun + np.cos(glonradv) * distv * np.cos(glatradv)
yposv = np.sin(glonradv) * distv * np.cos(glatradv)
zposv = distv * np.sin(glatradv) + zsun
# rgal with Reid et al. value
rgalv = np.sqrt(xposv**2 + yposv**2)

# plx, pmra, pmdec -> vx,vy,vz
# velocity
Tpmllpmbb = bovy_coords.pmrapmdec_to_pmllpmbb( \
    pmrav ,pmdecv ,rav ,decv, degree=True, epoch=None)
Tvxvyvz=bovy_coords.vrpmllpmbb_to_vxvyvz( \
    hrvv, Tpmllpmbb[:,0], Tpmllpmbb[:,1], glonv, glatv, distv, \
    XYZ=False, degree=True)
vxv = Tvxvyvz[:, 0] + usun
vyv = Tvxvyvz[:, 1] + vsun
vzv = Tvxvyvz[:, 2] + wsun
# Vcirc at the radius of the stars, including dVc/dR
vcircrv = vcirc + dvcdr * (rgalv - rsun)
vyv = vyv + vcircrv
# original velocity
vxv0 = vxv
vyv0 = vyv
vzv0 = vzv
# Galactic radius and velocities
vradv0 = (vxv0 * xposv + vyv0 * yposv) / rgalv
vrotv0 = (vxv0 * yposv - vyv0 * xposv) / rgalv
# then subtract circular velocity contribution
vxv = vxv - vcircrv * yposv / rgalv
vyv = vyv + vcircrv * xposv / rgalv
vradv = (vxv * xposv + vyv * yposv) / rgalv
vrotv = (vxv * yposv - vyv * xposv) / rgalv

# compute distance from the arm
# Perseus
angenr14 = (180.0 - 88.0) * np.pi / 180.0
angstr14 = (180.0 + 21.0) * np.pi / 180.0
angrefr14 = (180.0 - 14.2) * np.pi / 180.0
rrefr14 = 9.9
# pitchangle
tanpa = np.tan(9.4 * np.pi / 180.0)

# print ' ang range=',angstr14,angenr14

# adding the systematic error of 0.15 mag to errmodv
# errmodv=np.sqrt(errmodv**2+0.15**2)

# MC error sampling
nmc = 10000
nmc = 10
if MCsample_v == False and MCsample_vgalp == False:
    nmc = 1

if myrank == 0:
    print ' nmc=', nmc

# parallel MC
rindx = range(myrank, nstarv, nprocs)
nsrank = len(plxv[rindx])

# sample from parallax, proper-motion covariance matrix
if MCsample_v == True:
    plxpmradec_mc = np.empty((nsrank, 3, nmc))
    plxpmradec_mc[:, 0, :] = np.atleast_2d(plxv[rindx]).T
    plxpmradec_mc[:, 1, :] = np.atleast_2d(pmrav[rindx]).T
    plxpmradec_mc[:, 2, :] = np.atleast_2d(pmdecv[rindx]).T
    for ii in range(nsrank):
        ip = rindx[ii]
        # constract covariance matrix
        tcov = np.zeros((3, 3))
        # /2 because of symmetrization below
        tcov[0, 0] = errplxv[ip]**2.0 / 2.0
        tcov[1, 1] = errpmrav[ip]**2.0 / 2.0
        tcov[2, 2] = errpmdecv[ip]**2.0 / 2.0
        tcov[0, 1] = plxpmra_corrv[ip] * errplxv[ip] * errpmrav[ip]
        tcov[0, 2] = plxpmdec_corrv[ip] * errplxv[ip] * errpmdecv[ip]
        tcov[1, 2] = pmradec_corrv[ip] * errpmrav[ip] * errpmdecv[ip]
        # symmetrise
        tcov = (tcov + tcov.T)
        # Cholesky decomp.
        L = np.linalg.cholesky(tcov)
        plxpmradec_mc[ii] += np.dot(L, np.random.normal(size=(3, nmc)))

# distribution of velocity and distance.
# -> pml pmb
    ratile = np.tile(rav[rindx], (nmc, 1)).flatten()
    dectile = np.tile(decv[rindx], (nmc, 1)).flatten()
    pmllbb_sam = bovy_coords.pmrapmdec_to_pmllpmbb( \
        plxpmradec_mc[:, 1, :].T.flatten(), \
        plxpmradec_mc[:, 2, :].T.flatten(), \
        ratile, dectile, degree=True, epoch=None)
# reshape
    pmllbb_sam = pmllbb_sam.reshape((nmc, nsrank, 2))
# distance MC sampling
    plxv_sam = plxpmradec_mc[:, 0, :].T
    # check negative parallax
    plxv_samflat= plxv_sam.flatten()
    copysamflat=np.copy(plxv_samflat)
    plxlim=0.001
    if len(copysamflat[plxv_samflat<plxlim])>0: 
        if myrank == 0: 
            print len(copysamflat[plxv_samflat<plxlim]),' plx set to ',plxlim
    plxv_samflat[copysamflat<plxlim]=plxlim
    plxv_sam = np.reshape(plxv_samflat,(nmc,nsrank))
    # distance
    distv_sam = 1.0/plxv_sam

#    f = open('mcsamp_plx.asc', 'w')
#    fs = open('mcsamp_plx_stats.asc', 'w')
#    for j in range(nstarv):
#        for i in range(nmc):
#            print >>f, "%d %d %f %f" % (i, j, plxv_sam[i, j], plxv[j])
#        print >>fs,"%d %f %f" % (j, errplxv[j], np.std(plxv_sam[:, j]))
#    fs.close()
#    f.close()
# correlation check
#    f = open('mcsamp_plxpm10.asc', 'w')
#    j = 10
#    for i in range(nmc):
#        print >>f, "%d %d %f %f %f %f %f %f" \
#            % (i, j, plxv_sam[i, j], plxv[j], \
#               plxpmradec_mc[j, 1, i],plxpmradec_mc[j, 2, i], \
#               pmrav[j], pmdecv[j])
#    f.close()

# radial velocity MC sampling
    hrvv_sam = np.random.normal(hrvv[rindx], errhrvv[rindx], (nmc, nsrank))
# -> vx,vy,vz
    vxvyvz_sam = bovy_coords.vrpmllpmbb_to_vxvyvz(hrvv_sam.flatten(), \
        pmllbb_sam[:, :, 0].flatten(), pmllbb_sam[:, :, 1].flatten(), \
        np.tile(glonv[rindx], (nmc, 1)).flatten(), \
        np.tile(glatv[rindx], (nmc, 1)).flatten(), \
        distv_sam.flatten(), degree=True)
# reshape
    vxvyvz_sam = vxvyvz_sam.reshape((nmc, nsrank, 3))

    # sampling the Galactic parameters
    if MCsample_vgalp == True:
        rsun_sam = np.tile(np.random.normal(rsun, rsunsig, nmc), (nsrank, 1)).T
        zsun_sam = np.tile(np.random.normal(zsun, zsunsig, nmc), (nsrank, 1)).T
        omgsun_sam = np.tile(np.random.normal(
            omgsun, omgsunsig, nmc), (nsrank, 1)).T
        dvcdr_sam = np.tile(np.random.normal(
            dvcdr, dvcdrsig, nmc), (nsrank, 1)).T
        usun_sam = np.tile(np.random.normal(usun, usunsig, nmc), (nsrank, 1)).T
        vsun_sam = np.tile(np.random.normal(vsun, vsunsig, nmc), (nsrank, 1)).T
        wsun_sam = np.tile(np.random.normal(wsun, wsunsig, nmc), (nsrank, 1)).T
    else:
        rsun_sam = np.tile(rsun, (nmc, nsrank))
        zsun_sam = np.tile(zsun, (nmc, nsrank))
        omgsun_sam = np.tile(omgsun, (nmc, nsrank))
        dvcdr_sam = np.tile(dvcdr, (nmc, nsrank))
        usun_sam = np.tile(usun, (nmc, nsrank))
        vsun_sam = np.tile(vsun, (nmc, nsrank))
        wsun_sam = np.tile(wsun, (nmc, nsrank))

#    f = open('mcgalpsamp.asc', 'w')
#    for j in range(nstarv):
#        for i in range(nmc):
#            print >>f, "%d %d %f %f %f %f %f" % (
#                i, j, rsun_sam[i, j], omgsun_sam[i, j], dvcdr_sam[i, j], usun_sam[i, j], vsun_sam[i, j])
#    f.close()

    vxv_sam = vxvyvz_sam[:, :, 0] + usun_sam
    vyv_sam = vxvyvz_sam[:, :, 1] + vsun_sam
    vzv_sam = vxvyvz_sam[:, :, 2] + wsun_sam
# x, y position
    glonradv_sam = np.tile(glonradv[rindx], (nmc, 1))
    glatradv_sam = np.tile(glatradv[rindx], (nmc, 1))
    xposv_sam = -rsun_sam + np.cos(glonradv_sam) * \
        distv_sam * np.cos(glatradv_sam)
    yposv_sam = np.sin(glonradv_sam) * distv_sam * np.cos(glatradv_sam)
    zposv_sam = distv_sam * np.sin(glatradv_sam) + zsun_sam
# rgal with Reid et al. value
    rgalv_sam = np.sqrt(xposv_sam**2 + yposv_sam**2)
# Vcirc at the radius of the stars, including dVc/dR
    vcircrv_sam = omgsun_sam * rsun_sam - \
        vsun_sam + dvcdr_sam * (rgalv_sam - rsun_sam)
    vyv_sam = vyv_sam + vcircrv_sam
# original velocity
    vxv0_sam = vxv_sam
    vyv0_sam = vyv_sam
    vzv0_sam = vzv_sam
# Galactic radius and velocities
    vradv0_sam = (vxv0_sam * xposv_sam + vyv0_sam * yposv_sam) / rgalv_sam
    vrotv0_sam = (vxv0_sam * yposv_sam - vyv0_sam * xposv_sam) / rgalv_sam
# then subtract circular velocity contribution
    vxv_sam = vxv_sam - vcircrv_sam * yposv_sam / rgalv_sam
    vyv_sam = vyv_sam + vcircrv_sam * xposv_sam / rgalv_sam
    vradv_sam = (vxv_sam * xposv_sam + vyv_sam * yposv_sam) / rgalv_sam
    vrotv_sam = (vxv_sam * yposv_sam - vyv_sam * xposv_sam) / rgalv_sam
else:
    nmc = 1
    if myrank == 0:
        print ' No MC sample, set nmc = 1'
    xposv_sam = np.tile(xposv[rindx], (nmc, 1))
    yposv_sam = np.tile(yposv[rindx], (nmc, 1))
    distv_sam = np.tile(distv[rindx], (nmc, 1))
    vradv_sam = np.tile(vradv[rindx], (nmc, 1))
    vrotv_sam = np.tile(vrotv[rindx], (nmc, 1))
    vzv_sam = np.tile(vzv[rindx], (nmc, 1))
    glonradv_sam = np.tile(glonradv[rindx], (nmc, 1))
    glatradv_sam = np.tile(glatradv[rindx], (nmc, 1))
    rsun_sam = np.tile(rsun, (nmc, nsrank))
    zsun_sam = np.tile(zsun, (nmc, nsrank))

# output
# f = open('mcsample.asc', 'w')
# for j in range(nstarv):
#    for i in range(nmc):
#        print >>f, "%f %f %f %f %f %f %f %f" % (
#            xposv_sam[i, j], yposv_sam[i, j], xposv[j], yposv[j], vradv_sam[i, j], vrotv_sam[i, j], vradv[j], vrotv[j])
# f.close()

# compute distances from the arm
# recompute angref and rref
# x,y arm ref position w.r.t. the Sun in Reid et al. (2014)
xarms = rrefr14 * np.cos(angrefr14) + rsunr14
yarms = rrefr14 * np.sin(angrefr14)
if myrank == 0:
    print ' arm reference x,y position w.r.t. the Sun=', xarms, yarms, \
        np.arcsin(yarms / rrefr14)
rref_mean = np.sqrt(yarms**2 + (xarms - rsun)**2)
angref_mean = np.pi - np.arcsin(yarms / rref_mean)
if myrank == 0:
    print ' Rref and angref (mean)=', rref_mean, 180.0 - angref_mean * 180.0 / np.pi
# new angref and Ref
rref_sam = np.sqrt(yarms**2 + (xarms - rsun_sam)**2)
angref_sam = np.pi - np.arcsin(yarms / rref_sam)
# for test
# for i in range(nmc):
#  print ' Rref,Angrefr14,Rsam,Angsam0,1=',rrefr14,angrefr14,rref_sam[i,0] \
#   ,angref_sam[i,0],rref_sam[i,1],angref_sam[i,1]

# Serial version
# 
# darmv_sam = np.zeros_like(xposv_sam.flatten())
# angarmv_sam = np.zeros_like(xposv_sam.flatten())
#
# darmv_sam, angarmv_sam = distarm(np.tile(tanpa, (nmc, nstarv)).flatten(), angref_sam.flatten(), rref_sam.flatten(), xposv_sam.flatten(), yposv_sam.flatten())

darmv_sam = np.zeros_like(xposv_sam)
angarmv_sam = np.zeros_like(xposv_sam)

for j in range(nsrank):
    darmv_sam[:,j], angarmv_sam[:,j] = distarm(np.tile(tanpa,nmc), \
        angref_sam[:,j], rref_sam[:,j], xposv_sam[:,j], yposv_sam[:,j])

if myrank == 0: 
    print ' dist arm finished'
# reshape
darmv_sam = darmv_sam.reshape((nmc, nsrank))
angarmv_sam = angarmv_sam.reshape((nmc, nsrank))

# check the position relative to the arm
rspv_sam = np.exp(tanpa * (angarmv_sam - angref_sam)) * rref_sam
xarmp_sam = rspv_sam * np.cos(angarmv_sam)
yarmp_sam = rspv_sam * np.sin(angarmv_sam)
# use rsun_sam (not used in dAarm in Ceph-kin)
darmsunv_sam = np.sqrt((xarmp_sam + rsun_sam)**2 + yarmp_sam**2)
distxyv_sam = distv_sam * np.cos(glatradv_sam)
for j in range(nsrank):
    for i in range(nmc):
        if distxyv_sam[i, j] > darmsunv_sam[i, j] and \
                xposv_sam[i, j] < -rsun_sam[i, j]:
            darmv_sam[i, j] = -darmv_sam[i, j]

# mean value
darmv_mean = np.mean(darmv_sam, axis=0).reshape(nsrank)
angarmv_mean = np.mean(angarmv_sam, axis=0).reshape(nsrank)
vradv_mean = np.mean(vradv_sam, axis=0).reshape(nsrank)
vrotv_mean = np.mean(vrotv_sam, axis=0).reshape(nsrank)
vzv_mean = np.mean(vzv_sam, axis=0).reshape(nsrank)

# compute angular distance from the arm
dangdiffarmv_sam, angdiffarmv_sam = dangarm( \
    np.tile(tanpa, (nmc, nsrank)).flatten(), \
    angref_sam.flatten(), rref_sam.flatten(), xposv_sam.flatten(), \
    yposv_sam.flatten())
# reshape
dangdiffarmv_sam = dangdiffarmv_sam.reshape((nmc,nsrank))
angdiffarmv_sam = angdiffarmv_sam.reshape((nmc,nsrank))

# velocity
uradv_sam = -vradv_sam
# mean and std value
# velocity
uradv_rank_mean = np.mean(-vradv_sam, axis=0).reshape(nsrank)
uradv_rank_std = np.std(-vradv_sam, axis=0).reshape(nsrank)
vrotv_rank_mean = np.mean(vrotv_sam, axis=0).reshape(nsrank)
vrotv_rank_std = np.std(vrotv_sam, axis=0).reshape(nsrank)
vzv_rank_mean = np.mean(vzv_sam, axis=0).reshape(nsrank)
vzv_rank_std = np.std(vzv_sam, axis=0).reshape(nsrank)
# darm
darmv_rank_mean = np.mean(darmv_sam, axis=0).reshape(nsrank)
darmv_rank_std = np.std(darmv_sam, axis=0).reshape(nsrank)
dangdiffarmv_rank_mean=np.mean(dangdiffarmv_sam, axis=0).reshape(nsrank)
dangdiffarmv_rank_std=np.std(dangdiffarmv_sam, axis=0).reshape(nsrank)
angdiffarmv_rank_mean=np.mean(angdiffarmv_sam, axis=0).reshape(nsrank)
angdiffarmv_rank_std=np.std(angdiffarmv_sam, axis=0).reshape(nsrank)

if nprocs > 1:
    # MPI send and receive the mean and std 
    # uradv mean
    uradv_mean = gather_allindx(nstarv, uradv_rank_mean, rindx)
    # uradv_std
    uradv_std = gather_allindx(nstarv, uradv_rank_std, rindx)
    # vrotv mean
    vrotv_mean = gather_allindx(nstarv, vrotv_rank_mean, rindx)
    # vrotv_std
    vrotv_std = gather_allindx(nstarv, vrotv_rank_std, rindx)
    # vzv mean
    vzv_mean = gather_allindx(nstarv, vzv_rank_mean, rindx)
    # vzv_std
    vzv_std = gather_allindx(nstarv, vzv_rank_std, rindx)
    # darmv_mean
    darmv_mean = gather_allindx(nstarv, darmv_rank_mean, rindx)
    # darmv_std
    darmv_std = gather_allindx(nstarv, darmv_rank_std, rindx)
    # dangdiffarmv_mean
    dangdiffarmv_mean = gather_allindx(nstarv, dangdiffarmv_rank_mean, rindx)
    # dangdiffarmv_std
    dangdiffarmv_std = gather_allindx(nstarv, dangdiffarmv_rank_std, rindx)
    # angdiffarmv_mean
    angdiffarmv_mean = gather_allindx(nstarv, angdiffarmv_rank_mean, rindx)
    # dangdiffarmv_std
    angdiffarmv_std = gather_allindx(nstarv, angdiffarmv_rank_std, rindx)
else:
    uradv_mean = uradv_rank_mean
    uradv_std = uradv_rank_std
    vrotv_mean = vrotv_rank_mean
    vrotv_std = vrotv_rank_std
    vzv_mean = vzv_rank_mean
    vzv_std = vzv_rank_std
    darmv_mean = darmv_rank_mean
    darmv_std = darmv_rank_std
    dangdiffarmv_mean = dangdiffarmv_rank_mean
    dangdiffarmv_std = dangdiffarmv_rank_std
    angdiffarmv_mean = angdiffarmv_rank_mean
    angdiffarmv_std = angdiffarmv_rank_std

# sampling the stars around the arm
warm = 1.5
sindxarm =np.where((darmv_mean > -warm) & (darmv_mean < warm))
nswarm = np.size(sindxarm)
if myrank == 0:
    print ' sindxarm=', nswarm
sindxarm_list = np.asarray(sindxarm).flatten()

# output
if myrank == 0:
    ofile = 'GaiaDR2_RVS_'+glonlab+'.asc'
    f = open(ofile, 'w')
    print >>f, "# x y darm darm_sig Umean Usig Vmean Vsig darm_theta darm_theta_sig thetaarm thetaarm_sig Glon Glat Vzmean Vzsig z"
    for ii in range(nswarm):
        ip = sindxarm_list[ii]
        print >>f, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f" % (
            xposv[ip], yposv[ip], darmv_mean[ip], \
            darmv_std[ip], uradv_mean[ip], uradv_std[ip], \
            vrotv_mean[ip], vrotv_std[ip], dangdiffarmv_mean[ip], \
            dangdiffarmv_std[ip], angdiffarmv_mean[ip], angdiffarmv_std[ip], \
            glonv[ip],glatv[ip], vzv_mean[ip], 
            vzv_std[ip], zposv[ip])
    f.close()

# computing correlation coefficients
if nprocs == 1:
    ucorrcoef = np.zeros(nmc)
    vcorrcoef = np.zeros(nmc)
    for i in range(nmc):
        ucorrcoef[i] = np.corrcoef(darmv_sam[i, sindxarm], \
            uradv_sam[i, sindxarm])[0, 1]
        vcorrcoef[i] = np.corrcoef(darmv_sam[i, sindxarm], \
            vrotv_sam[i, sindxarm])[0, 1]

    if myrank == 0:
        print ' U corrcoef med,mean,sig=', np.median(ucorrcoef), \
            np.mean(ucorrcoef), np.std(ucorrcoef)
        print ' V corrcoef med,mean,sig=', np.median(vcorrcoef), \
            np.mean(vcorrcoef), np.std(vcorrcoef)
else:
    # no error analysis, array are too big
    ucorrcoef = np.corrcoef(darmv_mean[sindxarm], \
        uradv_mean[sindxarm])[0, 1]
    vcorrcoef = np.corrcoef(darmv_mean[sindxarm], \
        vrotv_mean[sindxarm])[0, 1]
    if myrank == 0:
        print ' U corrcoef =', ucorrcoef
        print ' V corrcoef =', vcorrcoef

# leading and training
# leading and training
# dleadmax = -0.2
# dleadmin = -1.5
# dtrailmax = 1.5
# dtrailmin = 0.2

dleadmax = -0.2
dleadmin = -1.0
dtrailmax = 1.0
dtrailmin = 0.2

sindxlead = np.where((darmv_mean > dleadmin) \
    & (darmv_mean < dleadmax))
sindxtrail = np.where((darmv_mean > dtrailmin) \
    & (darmv_mean < dtrailmax))
sindxlead_rank = np.where((darmv_mean[rindx] > dleadmin) \
    & (darmv_mean[rindx] < dleadmax))
sindxtrail_rank = np.where((darmv_mean[rindx] > dtrailmin) \
    & (darmv_mean[rindx] < dtrailmax))

if myrank == 0:
    print ' Leading d range min,max=', dleadmin, dleadmax
    print ' Trailing d range min,max=', dleadmin, dleadmax

nlead = np.size(sindxlead)
ntrail = np.size(sindxtrail)
nlead_rank = np.size(sindxlead_rank)
ntrail_rank = np.size(sindxtrail_rank)

if myrank == 0:
    print ' numper of stars leading=', nlead
    print ' numper of stars trailing=', ntrail

# U is positive toward centre
uradl_sam = -vradv_sam[:, sindxlead_rank].reshape(nmc, nlead_rank)
uradt_sam = -vradv_sam[:, sindxtrail_rank].reshape(nmc, ntrail_rank)
# V
vrotl_sam = vrotv_sam[:, sindxlead_rank].reshape(nmc, nlead_rank)
vrott_sam = vrotv_sam[:, sindxtrail_rank].reshape(nmc, ntrail_rank)

# mean values
uradt_rank_mean = np.mean(uradt_sam, axis=0).reshape(ntrail_rank)
uradt_rank_std = np.std(uradt_sam, axis=0).reshape(ntrail_rank)
vrott_rank_mean = np.mean(vrott_sam, axis=0).reshape(ntrail_rank)
vrott_rank_std = np.std(vrott_sam, axis=0).reshape(ntrail_rank)
uradl_rank_mean = np.mean(uradl_sam, axis=0).reshape(nlead_rank)
uradl_rank_std = np.std(uradl_sam, axis=0).reshape(nlead_rank)
vrotl_rank_mean = np.mean(vrotl_sam, axis=0).reshape(nlead_rank)
vrotl_rank_std = np.std(vrotl_sam, axis=0).reshape(nlead_rank)

if nprocs > 1:
    # MPI send and receive the mean and std 
    uradt_mean = gather_allindx(nstarv, uradt_rank_mean, rindx)
    uradt_std = gather_allindx(nstarv, uradt_rank_std, rindx)
    vrott_mean = gather_allindx(nstarv, vrott_rank_mean, rindx)
    vrott_std = gather_allindx(nstarv, vrott_rank_std, rindx)
    uradl_mean = gather_allindx(nstarv, uradl_rank_mean, rindx)
    uradl_std = gather_allindx(nstarv, uradl_rank_std, rindx)
    vrotl_mean = gather_allindx(nstarv, vrotl_rank_mean, rindx)
    vrotl_std = gather_allindx(nstarv, vrotl_rank_std, rindx)
else:
    uradt_mean = uradt_rank_mean
    uradt_std = uradt_rank_std
    vrott_mean = vrott_rank_mean
    vrott_std = vrott_rank_std
    uradl_mean = uradl_rank_mean
    uradl_std = uradl_rank_std
    vrotl_mean = vrotl_rank_mean
    vrotl_std = vrotl_rank_std


if nprocs == 1:
    # computing mean and median U, V
    uradl_mean_sam = np.mean(uradl_sam, axis=1)
    uradt_mean_sam = np.mean(uradt_sam, axis=1)
    vrotl_mean_sam = np.mean(vrotl_sam, axis=1)
    vrott_mean_sam = np.mean(vrott_sam, axis=1)
    uradl_median_sam = np.median(uradl_sam, axis=1)
    uradt_median_sam = np.median(uradt_sam, axis=1)
    vrotl_median_sam = np.median(vrotl_sam, axis=1)
    vrott_median_sam = np.median(vrott_sam, axis=1)

    # taking statistical mean and dispersion
    if myrank == 0:
        print '### Statistical mean and dispersion for Mean'
        print ' Leading U mean,sig=', np.mean(uradl_mean_sam), \
            np.std(uradl_mean_sam)
        print ' Trailing U mean,mean,sig=', np.mean(uradt_mean_sam), \
            np.std(uradt_mean_sam)
        print ' Leading V mean,sig=', np.mean(vrotl_mean_sam), \
            np.std(vrotl_mean_sam)
        print ' Trailing V mean,sig=', np.mean(vrott_mean_sam), \
            np.std(vrott_mean_sam)
        print '### Statistical mean and dispersion for Median'
        print ' Leading U mean,sig=', np.mean(uradl_median_sam), \
            np.std(uradl_median_sam)
        print ' Trailing U mean,mean,sig=', np.mean(uradt_median_sam), \
            np.std(uradt_median_sam)
        print ' Leading V mean,sig=', np.mean(vrotl_median_sam), \
            np.std(vrotl_median_sam)
        print ' Trailing V mean,sig=', np.mean(vrott_median_sam), \
            np.std(vrott_median_sam)

    # vertex deviation
    # covariance of leading part
    # leading part
    # f=open('lvl_sam.asc','w')
    lvl_sam = np.zeros(nmc)
    for i in range(nmc):
        v2dl = np.vstack((uradl_sam[i, :], vrotl_sam[i, :]))
        vcovl = np.cov(v2dl)
        # vertex deviation
        lvl_sam[i] = (180.0 / np.pi) * 0.5 * np.arctan(2.0 * vcovl[0, 1] \
             / (vcovl[0, 0] - vcovl[1, 1]))
        if vcovl[1, 1] > vcovl[0, 0]:
            # if myrank == 0:
            #     print i, ' MC leading sigU>sigR,lv,sig_RV=', \
            #         vcovl[0, 0], vcovl[1, 1], lvl_sam[i], vcovl[0, 1]
            # Vorobyov & Theis (2006), under eq. 18
            lvl_sam[i] = lvl_sam[i] + 90.0 * np.sign(vcovl[0, 1])
            # if myrank == 0:
            #     print ' lv changed to ', lvl_sam[i]

#  print >>f,"%d %f %f %f %f" %(i,lvl_sam[i],vcovl[0,0],vcovl[1,1],vcovl[0,1])
# f.close()

    if myrank == 0:
        print '### Vertex deviation'
        print ' Leading vertex deviation mean,sig=', np.mean(lvl_sam), \
            np.std(lvl_sam)

    # trailing part
    lvt_sam = np.zeros(nmc)
    for i in range(nmc):
        v2dt = np.vstack((uradt_sam[i, :], vrott_sam[i, :]))
        vcovt = np.cov(v2dt)
        lvt_sam[i] = (180.0 / np.pi) * 0.5 * np.arctan(2.0 * vcovt[0, 1] \
            / (vcovt[0, 0] - vcovt[1, 1]))
        if vcovl[1, 1] > vcovl[0, 0]:
            # Vorobyov & Theis (2006), under eq. 18
            lvl_sam[i] = lvl_sam[i] + 90.0 * np.sign(vcovl[0, 1])

    if myrank == 0:
        print ' Trailing vertex deviation mean,sig=', np.mean(lvt_sam), \
            np.std(lvt_sam)

else:
    # taking statistical mean and dispersion
    if myrank == 0:
        print '### Statistical mean and dispersion for mean velocity'
        print ' Leading U mean,sig=', np.mean(uradl_mean), \
            np.std(uradl_mean)
        print ' Trailing U mean,mean,sig=', np.mean(uradt_mean), \
            np.std(uradt_mean)
        print ' Leading V mean,sig=', np.mean(vrotl_mean), \
            np.std(vrotl_mean)
        print ' Trailing V mean,sig=', np.mean(vrott_mean), \
            np.std(vrott_mean)
        print '### Statistical mean and dispersion for Median'
        print ' Leading U mean,sig=', np.mean(uradl_mean), \
            np.std(uradl_mean)
        print ' Trailing U mean,mean,sig=', np.mean(uradt_mean), \
            np.std(uradt_mean)
        print ' Leading V mean,sig=', np.mean(vrotl_mean), \
            np.std(vrotl_mean)
        print ' Trailing V mean,sig=', np.mean(vrott_mean), \
            np.std(vrott_mean)

    # vertex deviation
    # covariance of leading part
    # leading part
    # f=open('lvl_sam.asc','w')
    v2dl = np.vstack((uradl_mean, vrotl_mean))
    vcovl = np.cov(v2dl)
    # vertex deviation
    lvl = (180.0 / np.pi) * 0.5 * np.arctan(2.0 * vcovl[0, 1] \
         / (vcovl[0, 0] - vcovl[1, 1]))
    if vcovl[1, 1] > vcovl[0, 0]:
       # Vorobyov & Theis (2006), under eq. 18
       lvl = lvl + 90.0 * np.sign(vcovl[0, 1])
    # if myrank == 0:
    #     print ' lv changed to ', lvl_sam[i]

    if myrank == 0:
        print '### Vertex deviation'
        print ' Leading vertex deviation =',lvl

    # trailing part
    v2dt = np.vstack((uradt_mean, vrott_mean))
    vcovt = np.cov(v2dt)
    lvt = (180.0 / np.pi) * 0.5 * np.arctan(2.0 * vcovt[0, 1] \
            / (vcovt[0, 0] - vcovt[1, 1]))
    if vcovl[1, 1] > vcovl[0, 0]:
        # Vorobyov & Theis (2006), under eq. 18
        lvl = lvl + 90.0 * np.sign(vcovl[0, 1])

    if myrank == 0:
        print ' Trailing vertex deviation =', lvt

nrows = 3
ncols = 3
if myrank == 0:
    # plot d vs. U
    plt.subplot(nrows, ncols, 1)
    plt.errorbar(darmv_mean[sindxarm], uradv_mean[sindxarm],
        xerr=darmv_std[sindxarm], yerr=uradv_std[sindxarm], fmt='.', \
                 marker='.', ms=0.5)
    plt.xlabel(r"d (kpc)", fontsize=12, fontname="serif")
    plt.ylabel(r"U (km/s)", fontsize=12, fontname="serif")
    plt.axis([-warm, warm, -80.0, 80.0], 'scaled')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plot d vs. V
    # plt.subplot(gs1[1])
    plt.subplot(nrows, ncols, 2)
    plt.errorbar(darmv_mean, vrotv_mean, \
        xerr=darmv_std, yerr=vrotv_std, fmt='.', \
                 marker='.', ms=0.5)
    plt.xlabel(r"d (kpc)", fontsize=12, fontname="serif")
    plt.ylabel(r"V (km/s)", fontsize=12, fontname="serif")
    plt.axis([-warm, warm, -80.0, 80.0], 'scaled')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # bottom panel
    # U hist
    # gs2=gridspec.GridSpec(1,3)
    # gs2.update(left=0.1,right=0.975,bottom=0.1,top=0.4)
    # plt.subplot(gs2[0])
    plt.subplot(nrows, ncols, 4)
    plt.hist(uradt_mean.flatten(), bins=40, range=(-60, 80),
         normed=True, histtype='step', color='b', linewidth=2.0)
    plt.hist(uradl_mean.flatten(), bins=40, range=(-60, 80),
         normed=True, histtype='step', color='r', linewidth=2.0)
    plt.xlabel(r"U (km/s)", fontsize=12, fontname="serif")
    plt.ylabel(r"dN", fontsize=12, fontname="serif")
    # V hist
    # plt.subplot(gs2[1])
    plt.subplot(nrows, ncols, 5)
    plt.hist(vrott_mean, bins=40, range=(-60, 80),
         normed=True, histtype='step', color='b', linewidth=2.0)
    plt.hist(vrotl_mean, bins=20, range=(-60, 80),
         normed=True, histtype='step', color='r', linewidth=2.0)
    plt.xlabel(r"V (km/s)", fontsize=12, fontname="serif")
    plt.ylabel(r"dN", fontsize=12, fontname="serif")
    # U-V map
    # leading
    # plt.subplot(gs1[2])
    plt.subplot(nrows, ncols, 3)
    # plt.scatter(uradl,vrotl,s=30)

    plt.errorbar(uradl_mean, vrotl_mean, xerr=uradl_std, yerr=vrotl_std,
                 fmt='.', marker='.', ms=0.5)
    plt.xlabel(r"U (km/s)", fontsize=12, fontname="serif")
    plt.ylabel(r"V (km/s)", fontsize=12, fontname="serif")
    plt.axis([-80.0, 80.0, -80.0, 80.0], 'scaled')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # trailing
    # plt.subplot(gs2[2])
    plt.subplot(nrows, ncols, 6)
    # plt.scatter(uradt,vrott,s=30)
    plt.errorbar(uradt_mean, vrott_mean, xerr=uradt_std, yerr=vrott_std, \
                 fmt='.', marker='.', ms=0.5)
    plt.xlabel(r"U (km/s)", fontsize=12, fontname="serif")
    plt.ylabel(r"V (km/s)", fontsize=12, fontname="serif")
    plt.axis([-80.0, 80.0, -80.0, 80.0], 'scaled')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plot
    plt.tight_layout()
    plt.show()
    # plt.savefig('vdisp.png')
    plt.clf()
    plt.close()

comm.Barrier()
comm.Disconnect()
