
# Strategy

Following http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt pick up A5V-A9V sstars from LAMOST data. Cross-match them with Gaia DR2.

Teff range: 7330<Teff<8040.0

Age range: 8.8<Log Age<9.0, 630 Myr< Age < 1.0

Logg>3.2

Rv err<10.0

Distance<3.75 kpc (15% parallax error expected in DR2)

140<Glon<220 deg

|z|<0.5 kpc

# Mock data exercise todo



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

no velocity data read. 
