# DDRM

Information
-----------
Debris Disk Reflectance Modeling (DDRM) code for fitting disk reflectance spectrum measured by [JWST/NIRSpec IFU](https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph#gsc.tab=0). DDRM can also be used for fitting disk reflectance spectra measured by other instruments.  

DDRM will create a model reflectance and fit the measured disk reflectance spectrum using scipy.optimize.curve_fit and/or emcee. Curve_fit can provide a good fit and use it as initial input in the MCMC analysis (optional). 

The disk reflectance spectrum shows the scattering efficiency of dust particles in the disk. In DDRM, different dust species were mixed to form an effective medium following the Bruggeman rule. Mie theory was used to calculate the scattering efficiency of a given medium.

If you find a bug or want to suggest improvements, please [create a ticket](https://github.com/ChenXie-astro/DDRM/issues).


Requirements
------------
  - numpy
  - scipy
  - matplotlib
  - scipy
  - astropy
  - [emcee](https://emcee.readthedocs.io/en/stable/)
  - [miepython](https://miepython.readthedocs.io/en/latest/)
  - [mpmath](https://mpmath.org)
  - [corner](https://corner.readthedocs.io/en/latest/)


Optical constants
--------
It is possible to use custom optical constant files following the same data structure (wavelength, n, k) as in [the optical_constants folder](https://github.com/ChenXie-astro/DDRM/optical_constants) 

There are optical constant files for H2O, FeS, Olivine (MgFeSiO4), amorphous carbon, pyroxene (MgSiO3, Mg0.7Fe0.3SiO3, Mg0.4Fe0.6SiO3), and SiC.\
Relevant references are listed below:\
Water ice: [Mastrapa et al., 2009](https://ui.adsabs.harvard.edu/abs/2009ApJ...701.1347M/abstract)\
FeS: [Henning & Stognienko 1996](https://ui.adsabs.harvard.edu/abs/1996A&A...311..291H)\
Olivine: [Dorschner et al., 1995](https://ui.adsabs.harvard.edu/abs/1995A&A...300..503D)\
Amporphous carbon: [Preibisch 1993](https://ui.adsabs.harvard.edu/abs/1993A&A...279..577P)\
Pyroxene: [Dorschner et al., 1995](https://ui.adsabs.harvard.edu/abs/1995A&A...300..503D)\
SiC: [Laor & Draine 1993](https://ui.adsabs.harvard.edu/abs/1993ApJ...402..441L)


Examples
--------
Model parameters and adopted dust components in the dust model can be modified in each Python script. 

To fit the spectrum with a single dust population using curve_fit
```
% python spec_fitting_curvefit_single_population.py
```
To fit the spectrum with two dust populations using curve_fit
```
% python spec_fitting_curvefit_two_population.py
```
To fit the spectrum with two dust populations using mcmc
```
% python spec_fitting_mcmc_single_population.py
```
To fit the spectrum with two dust populations using mcmc
```
% python spec_fitting_mcmc_two_population.py
```

It will run DDRM, create a best-fit model, and output posterior distributions of dust parameters (using mcmc).
Note: depending on sampling and spectral coverage, it may take ~1 hour to run the curve_fit pipeline and >8 hours to run the mcmc pipeline on the personal laptop.

Credits
-------
Xie et al. **in prep.**
