# DDRM

Information
-----------
Debris Disk Reflectance Modeling code for fitting disk reflectance spectrum measured by NIRSpec IFU.
We fitted the measured disk reflectance spectrum with models of reflectance spectrum using scipy.optimize.curve_fit and the emcee package. Curve_fit can provide a good fit and use it as initial input in the MCMC analysis.

If you find a bug or want to suggest improvements, please [create a ticket](https://github.com/ChenXie-astro/DDRM/issues).

Requirements
------------
  - numpy
  - scipy
  - matplotlib
  - corner
  - scipy
  - astropy
  - [emcee](https://emcee.readthedocs.io/en/stable/)
  - [miepython](https://miepython.readthedocs.io/en/latest/)
  - [mpmath](https://mpmath.org)

Optical constants
--------
It is possible to use custom optical constants following the same data sturctions as in [this folder]()

Examples
--------
Model parameters and adopted dust componennts in the dust model can be modified in each python script. 

To fit the spectrum with single dust population using curve_fit
```
% python spec_fitting_curvefit_single_population.py
```
To fit the spectrum with two dust population using curve_fit
```
% python spec_fitting_curvefit_two_population.py
```
To fit the spectrum with two dust population using mcmc
```
% python spec_fitting_mcmc_single_population.py
```
To fit the spectrum with two dust population using mcmc
```
% python spec_fitting_mcmc_two_population.py
```

It will run DDRM, create a best-fit model (curve_fit/mcmc), and posterior distributions of dust parameters (mcmc).


Credits
-------
Xie et al. in prep.
