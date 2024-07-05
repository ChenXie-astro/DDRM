# DDRM

Information
-----------
Debris Disk Reflectance Modeling (DDRM) code for fitting disk reflectance spectrum measured by [JWST/NIRSpec IFU](https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph#gsc.tab=0). DDRM can also be used for fitting disk reflectance spectra measured by other instruments.  

DDRM will create a model reflectance and fit the measured disk reflectance spectrum using scipy.optimize.curve_fit and/or emcee. Curve_fit can provide a good fit and use it as initial input in the MCMC analysis (optional). 

The disk reflectance spectrum shows the scattering efficiency of dust particles in the disk. In DDRM, different dust species were mixed to form an effective medium following the Bruggeman rule. Mie theory was then used to calculate the scattering efficiency of a given medium.

If you find a bug or want to suggest improvements, please [create a ticket](https://github.com/ChenXie-astro/DDRM/issues).


Requirements
------------
  - numpy
  - scipy
  - matplotlib
  - [astropy](https://www.astropy.org)
  - [emcee](https://emcee.readthedocs.io/en/stable/)
  - [miepython](https://miepython.readthedocs.io/en/latest/)
  - [mpmath](https://mpmath.org)
  - [corner](https://corner.readthedocs.io/en/latest/)

Tested: Python 3.7.7, numpy 1.21.6, scipy 1.7.3, matplotlib 3.3.0, astropy 3.2.3, emcee 3.0rc1, miepython 2.5.3, mpmath 1.1.0, corner 2.1.0.

Optical constants
--------
It is easy to add custom optical constant files simply following the same data structure (wavelength, n, k) as in [the optical_constants folder](https://github.com/ChenXie-astro/DDRM/optical_constants).

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
Note: depending on sampling, spectral coverage, and initial input, it may take >1 hour to run the curve_fit pipeline and >8 hours to run the mcmc pipeline on a personal laptop.

Credits
-------
If you use DDRM in your research, please cite

Xie et al. **in prep.**

In addition:

If you use mpmath.findroot in solving the Bruggeman rule, please also cite
```
@manual{mpmath,
  key     = {mpmath},
  author  = {The mpmath development team},
  title   = {mpmath: a {P}ython library for arbitrary-precision floating-point arithmetic (version 1.3.0)},
  note    = {{\tt http://mpmath.org/}},
  year    = {2023},
}
```

If you use miepython to calculate the scattering efficiency, please also cite 
```
@software{prahl_2024_11135148,
  author       = {Prahl, Scott},
  title        = {{miepython: Pure python calculation of Mie 
                   scattering}},
  month        = may,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {2.5.4},
  doi          = {10.5281/zenodo.11135148},
  url          = {https://doi.org/10.5281/zenodo.11135148}
}
```

If you use corner to make the corner plot, please also cite
```
  @article{corner,
      doi = {10.21105/joss.00024},
      url = {https://doi.org/10.21105/joss.00024},
      year  = {2016},
      month = {jun},
      publisher = {The Open Journal},
      volume = {1},
      number = {2},
      pages = {24},
      author = {Daniel Foreman-Mackey},
      title = {corner.py: Scatterplot matrices in Python},
      journal = {The Journal of Open Source Software}
    }
```
