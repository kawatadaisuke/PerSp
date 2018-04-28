
#
# dVrotVrVz
#
# reading DR/*.fits
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
from extreme_deconvolution import extreme_deconvolution
from mpi4py import MPI

# for not displaying
matplotlib.use('Agg')

comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
myrank = comm.Get_rank()

##### main programme start here #####

# flags
# if including RVS data
RVS = False
# True: MC sampling on, otherwise no MC or XD
MCsample = False
# True: read star_error*.npy
FileErrors = False
# True: read gaussxd*.asc
FileGauXD = True
# True: output Gaussian model
FigGauMod = True

if FileGauXD == True:
    MCsample = False

# number of MC sampling
nmc = 100
# number of gaussian model
ngauss_sam=np.array([5, 3, 3])
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
zmaxlim = 0.5
glonlow = 0.0
glonhigh = 360.0
vloserrlim = 5.0
# minimum plx
plxlim=0.001

if myrank == 0:
    if MCsample == True:
        print ' MCsample is on. Nmc = ',nmc

# read data
# RVS data
# infilel0 = 'DR2/RVSl0-30z05-result.fits'
# infilel0 = 'DR2/RVSl60-90z05-result.fits'
# infilel0 = 'DR2/RVSl120-150z05-result.fits'
# infilel0 = 'DR2/RVSl180-210z05-result.fits'
# infilel0 = 'DR2/RVSl240-270z05-result.fits'
# infilel0 = 'DR2/RVSl300-330z05-result.fits'
# all
# infilel0 = 'DR2/MG152l0-30z05-result.fits'
# infilel0 = 'DR2/MG152l60-90z05-result.fits'
# infilel0 = 'DR2/MG152l120-150z05-result.fits'
# infilel0 = 'DR2/MG152l180-210z05-result.fits'
# infilel0 = 'DR2/MG152l240-270z05-result.fits'
infilel0 = 'DR2/MG152l300-330z05-result.fits'
starl0 = pyfits.open(infilel0)
# infilel1 = 'DR2/RVSl30-60z05-result.fits'
# infilel1 = 'DR2/RVSl90-120z05-result.fits'
# infilel1 = 'DR2/RVSl150-180z05-result.fits'
# infilel1 = 'DR2/RVSl210-240z05-result.fits'
# infilel1 = 'DR2/RVSl270-300z05-result.fits'
# infilel1 = 'DR2/RVSl330-360z05-result.fits'
# infilel1 = 'DR2/MG152l0-30z05-result.fits'
# infilel1 = 'DR2/MG152l90-120z05-result.fits'
# infilel1 = 'DR2/MG152l150-180z05-result.fits'
# infilel1 = 'DR2/MG152l210-240z05-result.fits'
# infilel1 = 'DR2/MG152l270-300z05-result.fits'
infilel1 = 'DR2/MG152l330-360z05-result.fits'
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

if myrank == 0:
    print ' sample number of stars =', len(star['parallax'])

gabsmag = star['phot_g_mean_mag'] \
    -(5.0*np.log10(100.0/np.fabs(star['parallax']))) \
    +star['a_g_val']
zabs = np.fabs((1.0/star['parallax']) \
    *np.sin(np.pi*star['b']/180.0)+zsun)
yabs = np.fabs((1.0/star['parallax']) \
    *np.sin(np.pi*star['l']/180.0))

# minimum distance limit
distmin = 0.0000000001

if RVS == True:
    sindx = np.where((zabs < zmaxlim) & 
        (star['parallax']>0.0) & (star['parallax']<1.0/distmin) & 
        (star['parallax_error']/star['parallax']<e_plxlim) & 
        (star['radial_velocity_error']>0.0) &
        (star['radial_velocity_error']<vloserrlim) &
        (np.logical_and(star['l']>glonlow,star['l']<glonhigh)))
else:
    sindx = np.where((zabs < zmaxlim) & 
        (star['parallax']>0.0) & (star['parallax']<1.0/distmin) & 
        (star['parallax_error']/star['parallax']<e_plxlim) & 
        (np.logical_and(star['l']>glonlow,star['l']<glonhigh)))

nstars = len(star['ra'][sindx])

if myrank == 0:
    print ' N selected=',nstars
# extract the stellar data
ras = star['ra'][sindx]
decs = star['dec'][sindx]
glons = star['l'][sindx]
glats = star['b'][sindx]
plxs_obs = star['parallax'][sindx]
pmras_obs = star['pmra'][sindx]
pmdecs_obs = star['pmdec'][sindx]
e_plxs = star['parallax_error'][sindx]
e_pmras = star['pmra_error'][sindx]
e_pmdecs = star['pmdec_error'][sindx]
# HRV
hrvs_obs = star['radial_velocity'][sindx]
e_hrvs = star['radial_velocity_error'][sindx]
# G, G_BP, G_RP
gmag_obs = star['phot_g_mean_mag'][sindx]
gbpmag_obs = star['phot_bp_mean_mag'][sindx]
grpmag_obs = star['phot_rp_mean_mag'][sindx]
# e_gmag = star['e_G'][sindx]
# e_gbpmag = star['e_G_BP'][sindx]
# e_grpmag = star['e_G_RP'][sindx]
# Teff
teff_obs = star['teff_val'][sindx]
# e_teff = star['e_Teff'][sindx]
# Av
av_obs = star['a_g_val'][sindx]
# error correalation
plxpmra_corrs = star['parallax_pmra_corr'][sindx]
plxpmdec_corrs = star['parallax_pmdec_corr'][sindx]
pmradec_corrs = star['pmra_pmdec_corr'][sindx]
# age [Fe/H] only for Galaxia
fehs_true = np.zeros_like(e_plxs)
ages_true = np.zeros_like(e_plxs)

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
rgals_obs = np.sqrt(xposgals_obs**2+yposgals_obs**2)

if RVS == True:
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

# set error zero
e_dists = np.zeros_like(dists_obs)
if RVS == True:
    e_vrots = np.zeros_like(vrads_obs)
    e_vrads = np.zeros_like(vrads_obs)
    e_vzs = np.zeros_like(vrads_obs)
else:
    e_vlons = np.zeros_like(vlons_obs)
    e_vlats = np.zeros_like(vlats_obs)

if MCsample == True:
    # sample from parallax proper motion covariance matrix
    # parallel MC
    rindx = range(myrank,nstars,nprocs)
    nsrank = len(plxs_obs[rindx])
    plxpmradec_mc = np.empty((nsrank, 3, nmc))
    plxpmradec_mc[:, 0, :] = np.atleast_2d(plxs_obs[rindx]).T
    plxpmradec_mc[:, 1, :] = np.atleast_2d(pmras_obs[rindx]).T
    plxpmradec_mc[:, 2, :] = np.atleast_2d(pmdecs_obs[rindx]).T
    for ii in range(nsrank):
        ip = rindx[ii]
        # constract covariance matrix
        tcov = np.zeros((3, 3))
        # /2 because of symmetrization below
        tcov[0, 0] = e_plxs[ip]**2.0 / 2.0
        tcov[1, 1] = e_pmras[ip]**2.0 / 2.0
        tcov[2, 2] = e_pmdecs[ip]**2.0 / 2.0
        tcov[0, 1] = plxpmra_corrs[ip] * e_plxs[ip] * e_pmras[ip]
        tcov[0, 2] = plxpmdec_corrs[ip] * e_plxs[ip] * e_pmdecs[ip]
        tcov[1, 2] = pmradec_corrs[ip] * e_pmras[ip] * e_pmdecs[ip]
        # symmetrise
        tcov = (tcov + tcov.T)
        # Cholesky decomp.
        L = np.linalg.cholesky(tcov)
        plxpmradec_mc[ii] += np.dot(L, np.random.normal(size=(3, nmc)))

        # distribution of velocity and distance.
        # -> pml pmb
        ratile = np.tile(ras[rindx], (nmc, 1)).flatten()
        dectile = np.tile(decs[rindx], (nmc, 1)).flatten()
        pmllbb_sam = bovy_coords.pmrapmdec_to_pmllpmbb( \
            plxpmradec_mc[:, 1, :].T.flatten(), \
            plxpmradec_mc[:, 2, :].T.flatten(), \
            ratile, dectile, degree=True)
        # reshape
        pmllbb_sam = pmllbb_sam.reshape((nmc, nsrank, 2))
        # distance MC sampling
        plxs_sam = plxpmradec_mc[:, 0, :].T
        # check negative parallax
        plxs_samflat= plxs_sam.flatten()
        copysamflat=np.copy(plxs_samflat)
        if len(copysamflat[plxs_samflat<plxlim])>0: 
            print len(copysamflat[plxs_samflat<plxlim]),' plx set to ',plxlim
        plxs_samflat[copysamflat<plxlim]=plxlim
        plxs_sam = np.reshape(plxs_samflat,(nmc,nsrank))
        # distance
        dists_sam = 1.0/plxs_sam
        # mas/yr -> km/s
        vlons_sam = pmvconst*pmllbb_sam[:,:,0]*dists_sam
        vlats_sam = pmvconst*pmllbb_sam[:,:,1]*dists_sam
        # galactic position
        distxys_sam = dists_sam*np.cos(glatrads[rindx])
        xpos_sam = distxys_sam*np.cos(glonrads[rindx])
        ypos_sam = distxys_sam*np.sin(glonrads[rindx])
        zpos_sam = dists_sam*np.sin(glatrads[rindx])
        rgals_sam = np.sqrt((xpos_sam-rsun)**2+ypos_sam**2)

        if RVS == True:
            # 3D velocity errors
            hrvs_sam = np.random.normal(hrvs_obs[rindx], \
                e_hrvs[rindx], (nmc, nsrank))
            vxvyvz_sam = bovy_coords.vrpmllpmbb_to_vxvyvz( \
                hrvs_sam.flatten(), pmllbb_sam[:,:,0].flatten(), \
                pmllbb_sam[:,:,1].flatten(), \
                np.tile(glons[rindx], (nmc, 1)).flatten(), \
                np.tile(glats[rindx], (nmc, 1)).flatten(), \
                dists_sam.flatten(), degree=True)
            vxvyvz_sam = vxvyvz_sam.reshape((nmc, nsrank, 3))
            vxs_sam = vxvyvz_sam[:,:,0]
            vys_sam = vxvyvz_sam[:,:,1]
            vzs_sam = vxvyvz_sam[:,:,2]+wsun
            # 2D velocity
            hrvxys_sam = hrvs_sam*np.cos(glatrads[rindx])
            vxgals_sam = vxs_sam+usun
            vygals_sam = vys_sam+vsun
            vrots_sam = (vxgals_sam*ypos_sam-vygals_sam*(xpos_sam-rsun)) \
                /rgals_sam
            vrads_sam = (vxgals_sam*(xpos_sam-rsun)+vygals_sam*ypos_sam) \
                /rgals_sam
            # f = open('mcsample_stars.asc','w')
            # for j in range(100000,100100):
            #    for i in range(nmc):
            #        print >>f, "%d %d %f %f %f %f %f %f" % (i, j, \
            #            plxs_sam[i,j], plxs_obs[j], \
            #            rgals_sam[i,j] , rgals_obs[j], \
            #            vrots_sam[i,j] , vrots_obs[j])
            # f.close()

        # uncertainty estimats dispersion (use observed one for mean value)
        # distance
        e_dists[rindx] = np.std(dists_sam, axis=0).reshape(nsrank)
        if nprocs > 1:
            ncom = len(e_dists)
            sendbuf = np.zeros(ncom,dtype=np.float64)
            sendbuf = e_dists
            recvbuf = np.zeros(ncom,dtype=np.float64)
            comm.Allreduce(sendbuf,recvbuf,op=MPI.SUM)
            e_dists = recvbuf
        if RVS == True:
            # rgals
            e_rgals[rindx] = np.std(rgals_sam, axis=0).reshape(nsrank)
            # vzs
            e_vzs[rindx] = np.std(vzs_sam, axis=0).reshape(nsrank)
            # vrots
            e_vrots[rindx] = np.std(vrots_sam, axis=0).reshape(nsrank)
            if nprocs > 1:
                # MPI
                # e_rgals
                ncom = len(e_rgals)
                sendbuf = np.zeros(ncom,dtype=np.float64)
                sendbuf = e_rgals
                recvbuf = np.zeros(ncom,dtype=np.float64)
                comm.Allreduce(sendbuf,recvbuf,op=MPI.SUM)
                e_rgals = recvbuf
                # e_vzs
                sendbuf = np.zeros(ncom,dtype=np.float64)
                sendbuf = e_vzs
                recvbuf = np.zeros(ncom,dtype=np.float64)
                comm.Allreduce(sendbuf,recvbuf,op=MPI.SUM)
                e_vzs = recvbuf
                # e_vrots
                sendbuf = np.zeros(ncom,dtype=np.float64)
                sendbuf = e_vrots
                recvbuf = np.zeros(ncom,dtype=np.float64)
                comm.Allreduce(sendbuf,recvbuf,op=MPI.SUM)
                e_vrots = recvbuf
        else:
            # no RVS
            # vlons
            e_vlons[rindx] = np.std(vlons_sam, axis=0).reshape(nsrank)
            # vrots
            e_vlats[rindx] = np.std(vlats_sam, axis=0).reshape(nsrank)
            if nprocs > 1:
                # MPI
                # e_vlons
                ncom = len(e_vlons)
                sendbuf = np.zeros(ncom,dtype=np.float64)
                sendbuf = e_vlons
                recvbuf = np.zeros(ncom,dtype=np.float64)
                comm.Allreduce(sendbuf,recvbuf,op=MPI.SUM)
                e_vlons = recvbuf
                # e_vlats
                ncom = len(e_vlats)
                sendbuf = np.zeros(ncom,dtype=np.float64)
                sendbuf = e_vlats
                recvbuf = np.zeros(ncom,dtype=np.float64)
                comm.Allreduce(sendbuf,recvbuf,op=MPI.SUM)
                e_vlats = recvbuf

    # save numpy data
    if RVS == True:
        f=np.savez('star_dRVrotVrVzunc_RVS.npz',dists_obs=dists_obs, \
            rgals_obs=rgals_obs, vrots_obs=vrots_obs, vrads_obs=vrads_obs, \
            vzs_obs=vzs_obs, e_dists=e_dists, e_rgals=e_rgals, e_vrads=e_vrads, \
            e_vrots=e_vrots, e_vzs=e_vzs )
    else:
        f=np.savez('star_dVlonVlat.npz',dists_obs=dists_obs, \
            vlons_obs=vlons_obs, vlats_obs=vlats_obs, \
            e_dists=e_dists, e_vlons=e_vlons, e_vlats=e_vlats)

if RVS == True:
    # Vrot defined w.r.t. solar velocity
    vrots_obs -= vcircsun

# output velocity dispersion of the sample
if myrank == 0:
    if RVS == True:
        print ' velocity dispersion Vrot, Vrad, Vz = ', \
            np.std(vrots_obs), np.std(vrads_obs), np.std(vzs_obs)
    else:
        print ' velocity dispersion Vlon, Vlat = ', \
            np.std(vlons_obs), np.std(vlats_obs)

# take off median for Vlon
if RVS == False:
    vlons_obs -= np.median(vlons_obs)

# minimum number of stars in each column
nsmin = 25
# set number of grid
ngridx = 40
ngridy = 40
# grid plot for dist vs. V
drange = np.array([0.0, 4.5])

# number of velocity components
if RVS == True:
    nvel = 3
    vrange = np.array([[-50, 50.0], [-50, 50.0], [-20.0, 20.0]])
    vticks = np.array([[-40.0, -20.0, 0.0, 20.0, 40.0], \
        [-40.0, -20.0, 0.0, 20.0, 40.0], [-10.0, 0.0, 10.0]])
else:
    nvel = 2
    vrange = np.array([[-50, 50.0], [-40, 40.0]])
    vticks = np.array([[-20.0, 0.0, 20.0], \
        [-10.0, 0.0, 10.0]])

for ivel in range(nvel):

    if RVS == True:
        if ivel == 0:
            vvals = vrots_obs
        elif ivel == 1:
            vvals = vrads_obs
        else:
            vvals = vzs_obs
    else:
        if ivel == 0:
            vvals = vlons_obs
        else:
            vvals = vlats_obs
    # 2D histogram 
    H, xedges, yedges = np.histogram2d(dists_obs, vvals, \
                        bins=(ngridx, ngridy), \
                        range=(drange, vrange[ivel]))
    # set x-axis (Rgal) is axis=1
    H = H.T
    # normalised by column
    # print ' hist = ',H
    # print ' np column = ',np.sum(H, axis=0)
    H[:, np.sum(H, axis=0)<nsmin] = 0.0
    H[:, np.sum(H, axis=0)>=nsmin] = H[:, np.sum(H, axis=0)>=nsmin] \
      / np.sum(H[:, np.sum(H, axis=0)>=nsmin], axis=0)
    # print ' normalised hist = ',H
    # plt.imshow(H, interpolation='gaussian', origin='lower', aspect='auto', \
    #    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    # plt.colorbar(im)
    # plt.show()

    if ivel == 0: 
        H_dV0 = np.copy(H)
        H_dV0_xedges = np.copy(xedges)
        H_dV0_yedges = np.copy(yedges)
    elif ivel == 1:
        H_dV1 = np.copy(H)
        H_dV1_xedges = np.copy(xedges)
        H_dV1_yedges = np.copy(yedges)
    else:
        H_dV2 = np.copy(H)
        H_dV2_xedges = np.copy(xedges)
        H_dV2_yedges = np.copy(yedges)

# Vz median
ndd = 30+1
lowd = 0.0
highd = 3.0

dd = (highd-lowd)/(ndd-1)
vzmed_dd = np.zeros(ndd)
dgrid_vz = np.zeros(ndd)
dlow = lowd
dhigh = lowd+dd

for ii in range(ndd):
    sindx = np.where((dists_obs>=dlow) & (dists_obs<dhigh))
    dgrid_vz[ii] = 0.5*(dlow+dhigh)
    if RVS == True:
        vzmed_dd[ii] = np.median(vzs_obs[sindx])
    else:
        vzmed_dd[ii] = np.median(vlats_obs[sindx])
    dlow += dd
    dhigh += dd

# Final plot
if myrank == 0:
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stixsans"
    plt.rcParams["font.size"] = 16

    # colour mapscale
    cmin = 0.0
    cmax = 0.07
    if RVS == True:
        f, (ax1, ax2, ax3) = plt.subplots(3, sharex = True, figsize=(8,8))
    else:
        f, (ax1, ax2) = plt.subplots(2, sharex = True, figsize=(10,8))
    labpos = np.array([4.5, 40.0])
    ax1.imshow(H_dV0, interpolation='gaussian', origin='lower', \
           aspect='auto', vmin=cmin, vmax=cmax, \
           extent=[H_dV0_xedges[0], H_dV0_xedges[-1], \
                   H_dV0_yedges[0], H_dV0_yedges[-1]], \
           cmap=cm.jet)
    ax1.set_xlim(H_dV0_xedges[0], H_dV0_xedges[-1])
    ax1.set_ylim(H_dV0_yedges[0], H_dV0_yedges[-1])
    if RVS == True:
        ax1.set_ylabel(r"$V_{rot}$ (km s$^{-1}$)", fontsize=18)
    else:
        ax1.set_ylabel(r"$V_{lon}$ (km s$^{-1}$)", fontsize=18)
    ax1.tick_params(labelsize=16, color='k')
    ax1.set_yticks(vticks[0])

    # colour mapscale
    cmin = 0.0
    cmax = 0.07
    im = ax2.imshow(H_dV1, interpolation='gaussian', origin='lower', \
           aspect='auto', vmin=cmin, vmax=cmax, \
           extent=[H_dV1_xedges[0], H_dV1_xedges[-1], \
                   H_dV1_yedges[0], H_dV1_yedges[-1]], \
           cmap=cm.jet)
    ax2.set_xlim(H_dV1_xedges[0], H_dV1_xedges[-1])
    ax2.set_ylim(H_dV1_yedges[0], H_dV1_yedges[-1])
    if RVS == True:
        ax2.set_ylabel(r"$V_{\rm rad}$ (km s$^{-1}$)", fontsize=18)
    else:
        # median values
        # ax2.plot(dgrid_vz,vzmed_dd)
        ax2.set_ylabel(r"$V_{\rm b}$ (km s$^{-1}$)", fontsize=18)
    ax2.tick_params(labelsize=16, color='k')
    ax2.set_yticks(vticks[1])

    if RVS == True:
        im = ax3.imshow(H_dV2, interpolation='gaussian', origin='lower', \
           aspect='auto', vmin=cmin, vmax=cmax, \
           extent=[H_dV2_xedges[0], H_dV2_xedges[-1], \
                   H_dV2_yedges[0], H_dV2_yedges[-1]], \
           cmap=cm.jet)
        ax3.set_xlim(H_dV2_xedges[0], H_dV2_xedges[-1])
        ax3.set_ylim(H_dV2_yedges[0], H_dV2_yedges[-1])
        # median values
        ax3.plot(dgrid_vz,vzmed_dd)
        ax3.set_ylabel(r"$V_{\rm z}$ (km s$^{-1}$)", fontsize=18)
        ax3.tick_params(labelsize=16, color='k')
        ax3.set_yticks(vticks[2])

    plt.xlabel(r"D (kpc)", fontsize=18)
    f.subplots_adjust(hspace=0.0, right = 0.8)
    cbar_ax = f.add_axes([0.8, 0.15, 0.05, 0.7])
    cb = f.colorbar(im, cax=cbar_ax)
    cb.ax.tick_params(labelsize=16)
    plt.show()
    # plt.savefig('RVrot.eps')
    plt.close(f)
