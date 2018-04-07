
#
# darm_Astats.py
#
#  read gaia-LAMOST-Astars_mock.fits and analyse velocity properties around the Perseus Arm
#

import pyfits
import math
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from scipy import stats
from galpy.util import bovy_coords
from galpy.util import bovy_plot

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
MCsample_v = True
MCsample_vgalp = True

if MCsample_vgalp == True:
    MCsample_v = True

print ' MCsample_v, MCsample_vgalp=', MCsample_v, MCsample_vgalp

# Reid et al. (2014)'s Galactic parameters
# Sun's position
rsunr14 = 8.34

print ' Rsun in Reid et al. (2014)=', rsunr14

# Galactic parameters and uncertainties
# Bland-Hawthorn & Gerhard (2016)
rsun = 8.2
rsunsig = 0.1
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
dvcdr = -2.4
dvcdrsig = 1.2
# Vcirc
vcirc = omgsun * rsun
pmvconst = 4.74047

print '### Assumed Galactic parameters'
print ' Rsun=', rsun, '+-', rsunsig, ' kpc'
print ' Omega_sun=', omgsun, '+-', omgsunsig, ' km/s/kpc'
print ' Solar proper motion U =', usun, '+-', usunsig, ' km/s'
print ' Solar proper motion V =', vsun, '+-', vsunsig, ' km/s'
print ' Solar proper motion W =', wsun, '+-', wsunsig, ' km/s'
print ' dVc/dR =', dvcdr, '+-', dvcdrsig, ' km/s/kpc'
print ' default Vcirc=', vcirc, ' km/s'

if MCsample_vgalp == False:
    print ' Because the arm position will not be adjusted, rsun set to be Rr14=', rsunr14
    # rsun=rsunr14

# read the data with velocity with MC error
# read verr_mc.py output
infile = 'gaia-LAMOST-Astars_mock.fits'
star_hdus = pyfits.open(infile)
star = star_hdus[1].data
star_hdus.close()

# select stars with e_Plx/Plx_obs<0.15
# select only velocity error is small enough
Verrlim = 20.0
zmaxlim = 0.5
fPlxlim = 0.15
distmaxlim = 10.0
Plx_obs=star['Plx_obs']
e_Plx=star['e_Plx']
pmRA_obs=star['pmRA_obs']
e_pmRA=star['e_pmRA']
pmDEC_obs=star['pmDEC_obs']
e_pmDEC=star['e_pmDEC']

zwerr = (1.0 / (Plx_obs-e_Plx) ) * np.sin(star['Glat_true'] * np.pi / 180.0)
# from Hunt et al. (2014)
errVra = pmvconst * np.sqrt( (e_pmRA**2+( (pmRA_obs*e_pmRA/Plx_obs)**2 ) ) \
  / Plx_obs**2)
errVdec = pmvconst*np.sqrt((e_pmDEC**2+((pmDEC_obs*e_pmDEC/Plx_obs)**2)) \
  /Plx_obs**2)
verr = np.sqrt(errVra**2 + errVdec**2 + star['e_HRV']**2)
dist = 1.0/Plx_obs
sindx = np.where( (Plx_obs>0.0) & (e_Plx/Plx_obs<fPlxlim) &
                  (verr < Verrlim) & (np.abs(zwerr) < zmaxlim) &
                  (dist < distmaxlim))

# extract the necessary particle info
glonv = star['GLON_true'][sindx]
glatv = star['GLAT_true'][sindx]
# number of data points
nstarv = len(glatv)
print ' number of stars selected=', nstarv
plxv = star['Plx_obs'][sindx]
errplxv = star['e_Plx'][sindx]
distv = 1.0/plxv
# RA, DEC from Gaia data
rav = star['RA_obs'][sindx]
decv = star['DEC_obs'][sindx]
pmrav = star['pmRA_obs'][sindx]
pmdecv = star['pmDEC_obs'][sindx]
errpmrav = star['e_pmRA'][sindx]
errpmdecv = star['e_pmDEC'][sindx]
plxpmra_corrv = star['PARALLAX_PMRA_CORR']
plxpmdec_corrv = star['PARALLAX_PMDEC_CORR']
pmradec_corrv = star['PMRA_PMDEC_CORR'][sindx]

hrvv = star['HRV_obs'][sindx]
errhrvv = star['e_HRV'][sindx]
# radian glon and glat
glonradv = glonv * np.pi / 180.0
glatradv = glatv * np.pi / 180.0

# x, y position
xposv = -rsun + np.cos(glonradv) * distv * np.cos(glatradv)
yposv = np.sin(glonradv) * distv * np.cos(glatradv)
zposv = distv * np.sin(glatradv)
# rgal with Reid et al. value
rgalv = np.sqrt(xposv**2 + yposv**2)

# plx, pmra, pmdec -> vx,vy,vz
# velocity
Tpmllpmbb = bovy_coords.pmrapmdec_to_pmllpmbb( \
    pmrav ,pmdecv ,rav ,decv, degree=True, epoch=2000.0)
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
nmc = 1000
if MCsample_v == False and MCsample_vgalp == False:
    nmc = 1

print ' nmc=', nmc

# sample from parallax, proper-motion covariance matrix
if MCsample_v == True:
    pmradec_mc = np.empty((nstarv, 2, nmc))
    pmradec_mc[:, 0, :] = np.atleast_2d(pmrav).T
    pmradec_mc[:, 1, :] = np.atleast_2d(pmdecv).T
    for ii in range(nstarv):
        # constract covariance matrix
        tcov = np.zeros((3, 3))
        # /2 because of symmetrization below
        tcov[0, 0] = errplxv[ii]**2.0 / 2.0
        tcov[1, 1] = errpmrav[ii]**2.0 / 2.0
        tcov[2, 2] = errpmdecv[ii]**2.0 / 2.0
        tcov[0, 1] = plxpmra_corrv[ii]**2.0 * errplxv[ii] * errpmrav[ii]
        tcov[0, 2] = plxpmdec_corrv[ii]**2.0 * errplxv[ii] * errpmdecv[ii]
        tcov[1, 2] = pmradec_corrv[ii] * errpmrav[ii] * errpmdecv[ii]
        # symmetrise
        tcov = (tcov + tcov.T)
        # Cholesky decomp.
        L = np.linalg.cholesky(tcov)
        pmradec_mc[ii] += np.dot(L, np.random.normal(size=(3, nmc)))[1:,:]

# distribution of velocity and distance.
# -> pml pmb
    ratile = np.tile(rav, (nmc, 1)).flatten()
    dectile = np.tile(decv, (nmc, 1)).flatten()
    pmllbb_sam = bovy_coords.pmrapmdec_to_pmllpmbb(pmradec_mc[:, 0, :].T.flatten(
    ), pmradec_mc[:, 1:].T.flatten(), ratile, dectile, degree=True, epoch=2000.0)
# reshape
    pmllbb_sam = pmllbb_sam.reshape((nmc, nstarv, 2))
# distance MC sampling
    plxv_sam = np.random.normal(plxv, errplxv, (nmc, nstarv))
    # check negative parallax
    plxv_samflat=plxv_sam.flatten()
    copysamflat=np.copy(plxv_samflat)
    plxlim=0.001
    if len(copysamflat[plxv_samflat<plxlim])>0: 
      print len(copysamflat[plxv_samflat<plxlim]),' plx set to ',plxlim
    plxv_samflat[copysamflat<plxlim]=plxlim
    plxv_sam = np.reshape(plxv_samflat,(nmc,nstarv))
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

# radial velocity MC sampling
    hrvv_sam = np.random.normal(hrvv, errhrvv, (nmc, nstarv))
# -> vx,vy,vz
    vxvyvz_sam = bovy_coords.vrpmllpmbb_to_vxvyvz(hrvv_sam.flatten(), pmllbb_sam[:, :, 0].flatten(), pmllbb_sam[:, :, 1].flatten(
    ), np.tile(glonv, (nmc, 1)).flatten(), np.tile(glatv, (nmc, 1)).flatten(), distv_sam.flatten(), degree=True)
# reshape
    vxvyvz_sam = vxvyvz_sam.reshape((nmc, nstarv, 3))

    # sampling the Galactic parameters
    if MCsample_vgalp == True:
        rsun_sam = np.tile(np.random.normal(rsun, rsunsig, nmc), (nstarv, 1)).T
        omgsun_sam = np.tile(np.random.normal(
            omgsun, omgsunsig, nmc), (nstarv, 1)).T
        dvcdr_sam = np.tile(np.random.normal(
            dvcdr, dvcdrsig, nmc), (nstarv, 1)).T
        usun_sam = np.tile(np.random.normal(usun, usunsig, nmc), (nstarv, 1)).T
        vsun_sam = np.tile(np.random.normal(vsun, vsunsig, nmc), (nstarv, 1)).T
        wsun_sam = np.tile(np.random.normal(wsun, wsunsig, nmc), (nstarv, 1)).T
    else:
        rsun_sam = np.tile(rsun, (nmc, nstarv))
        omgsun_sam = np.tile(omgsun, (nmc, nstarv))
        dvcdr_sam = np.tile(dvcdr, (nmc, nstarv))
        usun_sam = np.tile(usun, (nmc, nstarv))
        vsun_sam = np.tile(vsun, (nmc, nstarv))
        wsun_sam = np.tile(wsun, (nmc, nstarv))

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
    glonradv_sam = np.tile(glonradv, (nmc, 1))
    glatradv_sam = np.tile(glatradv, (nmc, 1))
    xposv_sam = -rsun_sam + np.cos(glonradv_sam) * \
        distv_sam * np.cos(glatradv_sam)
    yposv_sam = np.sin(glonradv_sam) * distv_sam * np.cos(glatradv_sam)
    zposv_sam = distv_sam * np.sin(glatradv_sam)
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
    xposv_sam = np.tile(xposv, (nmc, 1))
    yposv_sam = np.tile(yposv, (nmc, 1))
    distv_sam = np.tile(distv, (nmc, 1))
    vradv_sam = np.tile(vradv, (nmc, 1))
    vrotv_sam = np.tile(vrotv, (nmc, 1))
    glonradv_sam = np.tile(glonradv, (nmc, 1))
    glatradv_sam = np.tile(glatradv, (nmc, 1))
    rsun_sam = np.tile(rsun, (nmc, nstarv))

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
print ' arm reference x,y position w.r.t. the Sun=', xarms, yarms, np.arcsin(yarms / rrefr14)
rref_mean = np.sqrt(yarms**2 + (xarms - rsun)**2)
angref_mean = np.pi - np.arcsin(yarms / rref_mean)
print ' Rref and angref (mean)=', rref_mean, 180.0 - angref_mean * 180.0 / np.pi
# new angref and Ref
rref_sam = np.sqrt(yarms**2 + (xarms - rsun_sam)**2)
angref_sam = np.pi - np.arcsin(yarms / rref_sam)
# for test
# for i in range(nmc):
#  print ' Rref,Angrefr14,Rsam,Angsam0,1=',rrefr14,angrefr14,rref_sam[i,0] \
#   ,angref_sam[i,0],rref_sam[i,1],angref_sam[i,1]
darmv_sam = np.zeros_like(xposv_sam.flatten())
angarmv_sam = np.zeros_like(xposv_sam.flatten())
darmv_sam, angarmv_sam = distarm(np.tile(tanpa, (nmc, nstarv)).flatten(
), angref_sam.flatten(), rref_sam.flatten(), xposv_sam.flatten(), yposv_sam.flatten())
print ' dist arm finished'
# reshape
darmv_sam = darmv_sam.reshape((nmc, nstarv))
angarmv_sam = angarmv_sam.reshape((nmc, nstarv))

# check the position relative to the arm
rspv_sam = np.exp(tanpa * (angarmv_sam - angref_sam)) * rref_sam
xarmp_sam = rspv_sam * np.cos(angarmv_sam)
yarmp_sam = rspv_sam * np.sin(angarmv_sam)
darmsunv_sam = np.sqrt((xarmp_sam + rsun)**2 + yarmp_sam**2)
distxyv_sam = distv_sam * np.cos(glatradv_sam)
for j in range(nstarv):
    for i in range(nmc):
        if distxyv_sam[i, j] > darmsunv_sam[i, j] and xposv_sam[i, j] < -rsun:
            darmv_sam[i, j] = -darmv_sam[i, j]

# mean value
darmv_mean = np.mean(darmv_sam, axis=0).reshape(nstarv)
angarmv_mean = np.mean(angarmv_sam, axis=0).reshape(nstarv)
vradv_mean = np.mean(vradv_sam, axis=0).reshape(nstarv)
vrotv_mean = np.mean(vrotv_sam, axis=0).reshape(nstarv)

# compute angular distance from the arm
dangdiffarmv_sam, angdiffarmv_sam=dangarm(np.tile(tanpa, (nmc, nstarv)).flatten(), \
angref_sam.flatten(), rref_sam.flatten(), xposv_sam.flatten(), yposv_sam.flatten())
# reshape
dangdiffarmv_sam = dangdiffarmv_sam.reshape((nmc,nstarv))
angdiffarmv_sam = angdiffarmv_sam.reshape((nmc,nstarv))

# mean and std value
dangdiffarmv_mean=np.mean(dangdiffarmv_sam, axis=0).reshape(nstarv)
dangdiffarmv_std=np.std(dangdiffarmv_sam, axis=0).reshape(nstarv)
angdiffarmv_mean=np.mean(angdiffarmv_sam, axis=0).reshape(nstarv)

# output
# f = open('mcsample_darm.asc', 'w')
# for j in range(nstarv):
#    for i in range(nmc):
#        if MCsample_v == True:
#            print >>f, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f" % ( \
#                xposv_sam[i, j], yposv_sam[i, j], xposv[j], yposv[j], vradv_sam[i, j], \
#                vrotv_sam[i, j], vradv[j], vrotv[j], darmv_sam[i, j], darmv_mean[j], \
#                vradv_mean[j], vrotv_mean[j], rsun_sam[i, j], vcircrv_sam[i, j]n, \
#                usun_sam[i, j], rgalv_sam[i, j], rref_sam[i, j], angref_sam[i, j], \
#                dangdiffarmv_sam[i,j],dangdiffarmv_mean[j], \
#                np.sqrt(xposv_sam[i,j]**2+yposv_sam[i,j]**2))
#        else:
#            print >>f, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f" % (
#                xposv_sam[i, j], yposv_sam[i, j], xposv[j], yposv[j], vradv_sam[i, j], \
#                vrotv_sam[i, j], vradv[j], vrotv[j], darmv_sam[i, j], darmv_mean[j], \
#                vradv_mean[j], vrotv_mean[j],dangdiffarmv_sam[i,j], \
#                np.sqrt(xposv_sam[i,j]**2+yposv_sam[i,j]**2))
# f.close()

# sampling the stars around the arm
nrows = 3
ncols = 3
warm = 1.5
sindxarm = np.where((darmv_mean > -warm) & (darmv_mean < warm))
nswarm = np.size(sindxarm)
print ' sindxarm=', nswarm
uradwarm_sam = -vradv_sam[:, sindxarm].reshape(nmc, nswarm)
vrotwarm_sam = vrotv_sam[:, sindxarm].reshape(nmc, nswarm)
darmwarm_sam = darmv_sam[:, sindxarm].reshape(nmc, nswarm)
dangdiffwarm_sam = dangdiffarmv_sam[:, sindxarm].reshape(nmc, nswarm)
angdiffwarm_sam = angdiffarmv_sam[:, sindxarm].reshape(nmc, nswarm)

# mean value

print ' number of stars selected from d_mean<', warm, '=', nswarm
uradwarm_mean = np.mean(uradwarm_sam, axis=0).reshape(nswarm)
vrotwarm_mean = np.mean(vrotwarm_sam, axis=0).reshape(nswarm)
darmwarm_mean = np.mean(darmwarm_sam, axis=0).reshape(nswarm)
uradwarm_std = np.std(uradwarm_sam, axis=0).reshape(nswarm)
vrotwarm_std = np.std(vrotwarm_sam, axis=0).reshape(nswarm)
darmwarm_std = np.std(darmwarm_sam, axis=0).reshape(nswarm)

dangdiffwarm_mean = np.mean(dangdiffwarm_sam, axis=0).reshape(nswarm)
dangdiffwarm_std = np.std(dangdiffwarm_sam, axis=0).reshape(nswarm)
angdiffwarm_mean = np.mean(angdiffwarm_sam, axis=0).reshape(nswarm)
angdiffwarm_std = np.std(angdiffwarm_sam, axis=0).reshape(nswarm)

xposwarm = xposv[sindxarm]
yposwarm = yposv[sindxarm]
glonwarm = glonv[sindxarm]
glatwarm = glatv[sindxarm]
# output
f = open('CephmeanV_darm.asc', 'w')
print >>f, "# x y darm darm_sig Umean Usig Vmean Vsig darm_theta darm_theta_sig thetaarm thetaarm_sig Glon Glat"
for i in range(nswarm):
    print >>f, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f" % (
        xposwarm[i], yposwarm[i], darmwarm_mean[i], darmwarm_std[i], uradwarm_mean[i], \
        uradwarm_std[i], vrotwarm_mean[i], vrotwarm_std[i], \
        dangdiffwarm_mean[i],dangdiffwarm_std[i],angdiffwarm_mean[i],angdiffwarm_std[i], \
        glonwarm[i],glatwarm[i])
f.close()


# computing correlation coefficients
ucorrcoef = np.zeros(nmc)
vcorrcoef = np.zeros(nmc)
for i in range(nmc):
    ucorrcoef[i] = np.corrcoef(darmwarm_sam[i, :], uradwarm_sam[i, :])[0, 1]
    vcorrcoef[i] = np.corrcoef(darmwarm_sam[i, :], vrotwarm_sam[i, :])[0, 1]

print ' U corrcoef med,mean,sig=', np.median(ucorrcoef), np.mean(ucorrcoef), np.std(ucorrcoef)
print ' V corrcoef med,mean,sig=', np.median(vcorrcoef), np.mean(vcorrcoef), np.std(vcorrcoef)

# plot d vs. U
plt.subplot(nrows, ncols, 1)
# plt.scatter(darmwarm_mean,uradwarm_mean,s=30)
plt.errorbar(darmwarm_mean, uradwarm_mean,
             xerr=darmwarm_std, yerr=uradwarm_std, fmt='.')
plt.xlabel(r"d (kpc)", fontsize=12, fontname="serif")
plt.ylabel(r"U (km/s)", fontsize=12, fontname="serif")
plt.axis([-warm, warm, -80.0, 80.0], 'scaled')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plot d vs. V
# plt.subplot(gs1[1])
plt.subplot(nrows, ncols, 2)
# plt.scatter(darmwarm_mean,vrotwarm_mean,s=30)
plt.errorbar(darmwarm_mean, vrotwarm_mean,
             xerr=darmwarm_std, yerr=vrotwarm_std, fmt='.',)
plt.xlabel(r"d (kpc)", fontsize=12, fontname="serif")
plt.ylabel(r"V (km/s)", fontsize=12, fontname="serif")
plt.axis([-warm, warm, -80.0, 80.0], 'scaled')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# leading and training
# leading and training
dleadmax = -0.2
dleadmin = -1.5
dtrailmax = 1.5
dtrailmin = 0.2
sindxlead = np.where((darmv_mean > dleadmin) & (darmv_mean < dleadmax))
sindxtrail = np.where((darmv_mean > dtrailmin) & (darmv_mean < dtrailmax))

print ' Leading d range min,max=', dleadmin, dleadmax
print ' Trailing d range min,max=', dleadmin, dleadmax
nlead = np.size(sindxlead)
ntrail = np.size(sindxtrail)
print ' numper of stars leading=', nlead
print ' numper of stars trailing=', ntrail

# U is positive toward centre
uradl_sam = -vradv_sam[:, sindxlead].reshape(nmc, nlead)
uradt_sam = -vradv_sam[:, sindxtrail].reshape(nmc, ntrail)
# V
vrotl_sam = vrotv_sam[:, sindxlead].reshape(nmc, nlead)
vrott_sam = vrotv_sam[:, sindxtrail].reshape(nmc, ntrail)

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
print '### Statistical mean and dispersion for Mean'
print ' Leading U mean,sig=', np.mean(uradl_mean_sam), np.std(uradl_mean_sam)
print ' Trailing U mean,mean,sig=', np.mean(uradt_mean_sam), np.std(uradt_mean_sam)
print ' Leading V mean,sig=', np.mean(vrotl_mean_sam), np.std(vrotl_mean_sam)
print ' Trailing V mean,sig=', np.mean(vrott_mean_sam), np.std(vrott_mean_sam)
print '### Statistical mean and dispersion for Median'
print ' Leading U mean,sig=', np.mean(uradl_median_sam), np.std(uradl_median_sam)
print ' Trailing U mean,mean,sig=', np.mean(uradt_median_sam), np.std(uradt_median_sam)
print ' Leading V mean,sig=', np.mean(vrotl_median_sam), np.std(vrotl_median_sam)
print ' Trailing V mean,sig=', np.mean(vrott_median_sam), np.std(vrott_median_sam)

# vertex deviation
# covariance of leading part
# leading part
# f=open('lvl_sam.asc','w')
lvl_sam = np.zeros(nmc)
for i in range(nmc):
    v2dl = np.vstack((uradl_sam[i, :], vrotl_sam[i, :]))
    vcovl = np.cov(v2dl)
    # vertex deviation
    lvl_sam[i] = (180.0 / np.pi) * 0.5 * np.arctan(2.0 * vcovl[0, 1]
                                                   / (vcovl[0, 0] - vcovl[1, 1]))
    if vcovl[1, 1] > vcovl[0, 0]:
        print i, ' MC leading sigU>sigR,lv,sig_RV=', vcovl[0, 0], vcovl[1, 1], lvl_sam[i], vcovl[0, 1]
# Vorobyov & Theis (2006), under eq. 18
        lvl_sam[i] = lvl_sam[i] + 90.0 * np.sign(vcovl[0, 1])
        print ' lv changed to ', lvl_sam[i]


#  print >>f,"%d %f %f %f %f" %(i,lvl_sam[i],vcovl[0,0],vcovl[1,1],vcovl[0,1])
# f.close()


print '### Vertex deviation'
print ' Leading vertex deviation mean,sig=', np.mean(lvl_sam), np.std(lvl_sam)

# trailing part
lvt_sam = np.zeros(nmc)
for i in range(nmc):
    v2dt = np.vstack((uradt_sam[i, :], vrott_sam[i, :]))
    vcovt = np.cov(v2dt)
    lvt_sam[i] = (180.0 / np.pi) * 0.5 * np.arctan(2.0 * vcovt[0, 1]
                                                   / (vcovt[0, 0] - vcovt[1, 1]))
    if vcovl[1, 1] > vcovl[0, 0]:
        print i, ' MC trailing sigU>sigR,lv=', vcovl[0, 0], vcovl[1, 1], lvt_sam[i]
# Vorobyov & Theis (2006), under eq. 18
        lvl_sam[i] = lvl_sam[i] + 90.0 * np.sign(vcovl[0, 1])
        print ' lv changed to ', lvl_sam[i]

print ' Trailing vertex deviation mean,sig=', np.mean(lvt_sam), np.std(lvt_sam)

# bottom panel
# U hist
# gs2=gridspec.GridSpec(1,3)
# gs2.update(left=0.1,right=0.975,bottom=0.1,top=0.4)
# plt.subplot(gs2[0])
plt.subplot(nrows, ncols, 4)
plt.hist(uradt_sam.flatten(), bins=40, range=(-60, 80),
         normed=True, histtype='step', color='b', linewidth=2.0)
plt.hist(uradl_sam.flatten(), bins=40, range=(-60, 80),
         normed=True, histtype='step', color='r', linewidth=2.0)
plt.xlabel(r"U (km/s)", fontsize=12, fontname="serif")
plt.ylabel(r"dN", fontsize=12, fontname="serif")
# V hist
# plt.subplot(gs2[1])
plt.subplot(nrows, ncols, 5)
plt.hist(vrott_sam.flatten(), bins=40, range=(-60, 80),
         normed=True, histtype='step', color='b', linewidth=2.0)
plt.hist(vrotl_sam.flatten(), bins=20, range=(-60, 80),
         normed=True, histtype='step', color='r', linewidth=2.0)
plt.xlabel(r"V (km/s)", fontsize=12, fontname="serif")
plt.ylabel(r"dN", fontsize=12, fontname="serif")
# U-V map
# leading
# plt.subplot(gs1[2])
plt.subplot(nrows, ncols, 3)
# plt.scatter(uradl,vrotl,s=30)

plt.errorbar(np.mean(uradl_sam, axis=0).reshape(nlead), np.mean(vrotl_sam, axis=0).reshape(nlead), xerr=np.std(
    uradl_sam, axis=0).reshape(nlead), yerr=np.std(vrotl_sam, axis=0).reshape(nlead), fmt='.')
plt.xlabel(r"U (km/s)", fontsize=12, fontname="serif")
plt.ylabel(r"V (km/s)", fontsize=12, fontname="serif")
plt.axis([-80.0, 80.0, -80.0, 80.0], 'scaled')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# trailing
# plt.subplot(gs2[2])
plt.subplot(nrows, ncols, 6)
# plt.scatter(uradt,vrott,s=30)
plt.errorbar(np.mean(uradt_sam, axis=0).reshape(ntrail), np.mean(vrott_sam, axis=0).reshape(ntrail), xerr=np.std(
    uradt_sam, axis=0).reshape(ntrail), yerr=np.std(vrott_sam, axis=0).reshape(ntrail), fmt='.')
plt.xlabel(r"U (km/s)", fontsize=12, fontname="serif")
plt.ylabel(r"V (km/s)", fontsize=12, fontname="serif")
plt.axis([-80.0, 80.0, -80.0, 80.0], 'scaled')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plot
plt.tight_layout()
plt.show()

plt.hist(lvl_sam, bins=40, range=(-60, 60),
         normed=True, histtype='step', linewidth=2.0)
plt.show()


# plot

# set radial velocity arrow
i = 0
vxarr = np.zeros((2, nstarv))
vyarr = np.zeros((2, nstarv))
vxarr[0, :] = xposv
vyarr[0, :] = yposv
# set dx dy
vxarr[1, :] = vxv / 100.0
vyarr[1, :] = vyv / 100.0

# plot circle,
an = np.linspace(0, 2.0 * np.pi, 100)
rad = 7.0
i = 0
rad = 4.0
while i < 15:
    rad = rad + 0.5
    plt.plot(rad * np.cos(an), rad * np.sin(an), 'k:')
    i += 1
# plot arm position from Reid et al. 2014
# number of points
nsp = 100
isp = 0
plotsparm = True
if plotsparm == True:
    numsp = 3
else:
    numsp = 0
while isp < numsp:
    # angle in R14 is clock-wise start at the Sun at (0.0, Rsun)
    # convert to the one anti-clockwise starting from +x, y=0
    if isp == 0:
        # Scutum Arm
        angen = (180.0 - 3.0) * np.pi / 180.0
#    angen=(180.0+45.0)*np.pi/180.0
        angst = (180.0 - 101.0) * np.pi / 180.0
        angref = (180.0 - 27.6) * np.pi / 180.0
        rref = 5.0
# pitchangle
        tanpa = np.tan(19.8 * np.pi / 180.0)
    elif isp == 1:
        # Sagittarius Arm
        angen = (180.0 + 2.0) * np.pi / 180.0
#    angen=(180.0+45.0)*np.pi/180.0
        angst = (180.0 - 68.0) * np.pi / 180.0
        angref = (180.0 - 25.6) * np.pi / 180.0
        rref = 6.6
# pitchangle
        tanpa = np.tan(6.9 * np.pi / 180.0)
    else:
        # Perseus Arm
        angen = (180.0 - 88.0) * np.pi / 180.0
        angst = (180.0 + 21.0) * np.pi / 180.0
        angref = (180.0 - 14.2) * np.pi / 180.0
        rref = 9.9
# pitchangle
        tanpa = np.tan(9.4 * np.pi / 180.0)
# logarithmic spiral arm , log r= tan(pa) theta, in the case of anti-clockwise arm
    an = np.linspace(angst, angen, nsp)
    xsp = np.zeros(nsp)
    ysp = np.zeros(nsp)
    i = 0
    while i < nsp:
        rsp = np.exp(tanpa * (an[i] - angref)) * rref
        xsp[i] = rsp * np.cos(an[i])
        ysp[i] = rsp * np.sin(an[i])
        i += 1
    if isp == 0:
        plt.plot(xsp, ysp, 'b-')
    elif isp == 1:
        plt.plot(xsp, ysp, 'r-')
    else:
        plt.plot(xsp, ysp, 'g-')
        f = open('PerseusArm.asc', 'w')
        for i in range(nsp):
            print >>f, "%f %f" % (xsp[i], ysp[i])
        f.close()
    isp += 1

# velocity arrow
i = 0
while i < nstarv:
    # x,y,dx,dy
    plt.arrow(vxarr[0, i], vyarr[0, i], vxarr[1, i], vyarr[1, i],
              fc="k", ec="k", head_width=0.05, head_length=0.1)
    i += 1

# plot Cepheids data point
plt.scatter(-rsun, 0.0, marker="*", s=100, color='k')
plt.scatter(xposv, yposv, c=darmv_mean, s=30, vmin=-4.0, vmax=4.0)
plt.xlabel(r"X (kpc)", fontsize=18, fontname="serif")
plt.ylabel(r"Y (kpc)", fontsize=18, fontname="serif")
plt.axis([-13.0, -3.0, -4.5, 4.5], 'scaled')
cbar = plt.colorbar()
cbar.set_label(r'$\delta$[Fe/H]')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()
