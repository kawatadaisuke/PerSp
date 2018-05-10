
# xyzVrotVrVz.py (copy from dVrotVrVz.py)

 convert the data to x, y, z, Vrot, Vrad, Vz 


# dVrotVrVz.py (copy from RVrotVzDR2/py/RVrotVzall-mpimc.py )

 To plot distance vs Vrot, Vr, Vz from all RVS data (with quality cut)

# Strategy before DR2

Following http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt pick up A5V-A9V sstars from LAMOST data. Cross-match them with Gaia DR2.

Teff range: 7330<Teff<8040.0

Age range: 8.8<Log Age<9.0, 630 Myr< Age < 1.0

Logg>3.2

Rv err<10.0

Distance<3.75 kpc (15% parallax error expected in DR2)

140<Glon<220 deg

|z|<0.5 kpc

# Mock data exercise todo

> export DUST_DIR=/Users/dkawata/work/pops/mwdust/DUST_DATA

> python galmap-LAMOST-Astar.py

copy sels_rv.asc to gcdp-ana/lbsels/ini and move to gcdp-ana/lbsels

> lbsels

copy output/lbsels_targets_rv.dat to this directory and move back to this directory.

> python lbsels2ubgaiae.py

> cd ubgaiaerrors

> gaia_errors

> cd ..

> python ubgaiae2fits.py

> mpirun -np 2 python dArm_Astats.py

# Codes

## dArm_Astars.py
 analyse velocity properties around the Perseus Arm

## ubgaiae2fits.py
 Convert ubgaiaerrors output to a fits file. 

## ubgaiaerrors/
 Fortran code to calculate Gaia DR2 expected errors and mock observational data.

## lbsels2ubgaiae.py
 read lbsels_targets_rv.dat from gcdp-ana/lbsels and output ubgaiaerrors input.

## galmap-LAMOST-Astar.py
 analyse the distribution, proper motion and etc. 

 need to set DUST_DATA for mwdust

no velocity data read. 
