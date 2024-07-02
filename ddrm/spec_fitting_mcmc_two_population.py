#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import os
import math
import time
from astropy.io import fits
from scipy.interpolate import interp1d
import miepython
from mpmath import findroot
import emcee
from multiprocessing import Pool, cpu_count
from utils import plot_spectrum_errorbar, hgg_phase_function, inter_n_i, output_cornor_mcmc_result

# %% FUNCTIONS

def chi2(data, data_unc, model, lnlike = True):
    """Calculate the chi-squared value or log-likelihood for given data and model. 
    Note: if data_unc has values <= 0, they will be ignored and replaced by NaN.
    Input:  data: 2D array, observed data.
            data_unc: 2D array, uncertainty/noise map of the observed data.
            lnlike: boolean, if True, then the log-likelihood is returned.
    Output: chi2: float, chi-squared or log-likelihood value."""
    data_unc[np.where(data_unc <= 0)] = np.nan
    chi2 = np.nansum(((data-model)/data_unc)**2)

    print('reduced chi2: ', chi2/(np.count_nonzero(~np.isnan(data_unc))-len(x_all)))

    if lnlike:
        loglikelihood = -0.5*np.log(2*np.pi)*np.count_nonzero(~np.isnan(data_unc)) - 0.5*chi2 - np.nansum(np.log(data_unc))
        # -n/2*log(2pi) - 1/2 * chi2 - sum_i(log sigma_i) 
        return loglikelihood
    return chi2

   
def integrand(a, neff, z_wave, f,  p):
    """
    Calculating the model reflectance at a given wavelength
    """
    x = 2*np.pi*a/z_wave
    refrel = neff
    qext, qsca, qback, g = miepython.mie(refrel,x)
    return f*qsca*np.pi* a**2 * a**(p)
    

############################### For MCMC ###############################
def generate_model_spectrum_mcmc_two_pop(x_all):
    """
    For MCMC; two dust population
    model parameters:
    [0, 1,   2,    3,    4,    5,       6,     7,   8,   ... ...  ,  -2,  -1 ]
    [f, p, a_min, a_max, p2, a_min2, a_max2,  f_B, f_C,  ... ...  , f_B2, f_C2]
    """
    # dust population # 1
    scaling_factor = x_all[0] 
    p = x_all[1]
    a_min = x_all[2]
    a_max = x_all[3]
    # dust population # 2 only has two dust species # i.e., [p2, a_min2, a_max2, f_B2, f_C2] and porosity = 1-f_B2 - f_C2
    p2 = x_all[4] 
    a_min2 = x_all[5]
    a_max2 = x_all[6]  

    n_parameters = len(x_all)

    I = np.zeros((nz))
    I_nz_iter = np.zeros((nz_iter))
    neff_brug = np.zeros((nz_iter), dtype=complex)
    I_nz_iter2 = np.zeros((nz_iter))
    neff_brug2 = np.zeros((nz_iter), dtype=complex)

    # number of components in the dust population #1 starts in n_parameters - 7
    # dust population #1
    N = n_parameters - 7 - n_parameters_pop2
    porosity = 1- sum(x_all[n+7] for n in np.arange(0, N))
    if  porosity <0: 
        return I
    vf = np.concatenate((np.array([porosity]), np.array([x_all[n+7] for n in np.arange(0, N) ])), axis =0)

    # dust population #2
    N2 = n_parameters_pop2
    porosity2 = 1- sum(x_all[n] for n in np.arange(-1*N2, 0))
    if a_min2 >= a_max2 or porosity2<0:
        return I
    vf2 = np.concatenate((np.array([porosity2]), np.array([x_all[n] for n in np.arange(-1*N2, 0) ])), axis =0)

    for z in range(nz_iter):
        n_bg_test = findroot(lambda n_bg: sum((vf[n]*(n_array[n,z]**2 - (n_bg)**2)/(n_array[n,z]**2 + 2*(n_bg)**2)) for n in np.arange(0, vf.shape[0])), initial_guess , solver='muller')
        n_bg_real = n_bg_test.real 
        n_bg_imag = n_bg_test.imag                    
        neff_brug[z] = complex(n_bg_real, -1* n_bg_imag)

        x_a = np.logspace(np.log10(a_min), np.log10(a_max), num_integ)
        I_nz_iter[z] = np.trapz(integrand(x_a, neff_brug[z], wave_inter[z], f,  p), x_a)

        # large particles water ice
        n_bg_test2 =findroot(lambda n_bg: sum((vf2[n]*(n_array2[n,z]**2 - (n_bg)**2)/(n_array2[n,z]**2 + 2*(n_bg)**2)) for n in np.arange(0, vf2.shape[0])), initial_guess , solver='muller')
        n_bg_real2 = n_bg_test2.real 
        n_bg_imag2 = n_bg_test2.imag                    
        neff_brug2[z] = complex(n_bg_real2, -1* n_bg_imag2)

        x_a2 = np.logspace(np.log10(a_min2), np.log10(a_max2), num_integ)
        I_nz_iter2[z] = np.trapz(integrand(x_a2, neff_brug2[z], wave_inter[z], f,  p2), x_a2)

    f_I = interp1d(wave_inter, I_nz_iter+I_nz_iter2 )
    I = f_I(z_wave)* scaling_factor 

    return I


def lnprior(x_var):
    for i in range(len(x_var)):
        if x_var[i] <= bounds[0, i] or x_var[i] >= bounds[1, i]:
            return -np.inf
    return 0


def lnpost_mcmc(x_all):
    if lnprior(x_all) == -np.inf:
        return -np.inf

    model = generate_model_spectrum_mcmc_two_pop(x_all) 
    if np.count_nonzero(model) == 0:
        lnlike =  -np.inf
    else:
        lnlike = chi2(data, unc, model)

    return lnlike


def inspect_parameter(x_all):
    """
    For two dust populations
    """
    # check for individual input values
    for i in range(len(x_all)):
        if x_all[i] <= bounds[0, i] or x_all[i] >= bounds[1, i]:
            return -np.inf
    a_min2 = x_all[5]
    a_max2 = x_all[6]  

    n_parameters = len(x_all)
    n_parameters_pop2 = 2
    # number of components in the dust population #1 starts in n_parameters - 7
    # dust population #1
    N = n_parameters - 7 - n_parameters_pop2
    porosity = 1- sum(x_all[n+7] for n in np.arange(0, N))
    # dust population #2
    N2 = n_parameters_pop2
    porosity2 = 1- sum(x_all[n] for n in np.arange(-1*N2, 0))

    if  porosity <0 or a_min2 >= a_max2 or porosity2<0:
        return -np.inf

    return 1


############################### end for MCMC ############################### 
def output_best_fit_mcmc_result_two_pop(samples):
    # if n_parameters_pop1 == 2 and n_parameters_pop2 ==2:
    """
    two dust population
    model parameters:
    [0, 1,   2,    3,    4,    5,       6,     7,   8,   ... ...  ,  -2,  -1 ]
    [f, p, a_min, a_max, p2, a_min2, a_max2,  f_B, f_C,  ... ...  , f_B2, f_C2]
    """
    # dust population # 1
    if n_parameters_pop1 == 2 and n_parameters_pop2 ==2:    
        scaling_factor, p, a_min, a_max, p2, a_min2, a_max2, f_B, f_C,  f_B2, f_C2  = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))   
        porosity = np.ones((3)) - f_B - f_C  
        porosity2 = np.ones((3)) - f_B2 - f_C2

        mcmc_values = np.array([scaling_factor, p, a_min, a_max, p2, a_min2, a_max2, porosity, f_B, f_C, porosity2, f_B2, f_C2])

        best_fit_para = [scaling_factor[0], p[0], a_min[0], a_max[0], p2[0], a_min2[0], a_max2[0],   f_B[0], f_C[0],   f_B2[0], f_C2[0]] 

        model_bestfit = generate_model_spectrum_mcmc_two_pop(best_fit_para) 
        n_freedom = np.count_nonzero(~np.isnan(unc))-len( best_fit_para)
        if np.count_nonzero(model_bestfit) == 0:
            chi2_reduced =  np.inf
        else:
            chi2_reduced = chi2(data, unc, model_bestfit, lnlike = False)/n_freedom

        labels = [r'$f$', r'$p$', r'$a_{min}$', r'$a_{max}$', r'$p2$', r'$a_{min2}$', r'$a_{max2}$', 
                 r'v_{0}'.format(comp1_name), r'v_{0}'.format(comp2_name), 
                 r'v_{0}'.format(complarge_name), r'v_{0}'.format(complarge2_name)   ]

        stats = (f'$\chi^{2}_{{v}}$ = {chi2_reduced:.2f}\n'
                f'$f$ = {scaling_factor[0]:.3f} \n'
                f'$a_{{min}}$ = {a_min[0]:.2f} ($\mu$m)\n'
                f'$a_{{max}}$ = {a_max[0]:.2f} ($\mu$m)\n'
                f'$p$ = {p[0]:.2f}\n'
                f'porosity = {porosity[0]:.2f}\n'
                f'{comp1_name} ({ice_T_amp:.0f}K) = {f_B[0]:.2f}\n'
                f'{comp2_name} ({ice_T:.0f}K) = {f_C[0]:.2f}\n'
                f'$p2$ = {p2[0]:.2f}\n'
                f'porosity2 = {porosity2[0]:.2f}\n'
                f'{complarge_name} ({ice_T:.0f}K) = {f_B2[0]:.2f}\n'
                f'{complarge2_name}= {f_C2[0]:.2f}\n'
                f'$a_{{min2}}$ = {a_min2[0]:.2f} ($\mu$m)\n'
                f'$a_{{max2}}$ = {a_max2[0]:.2f} ($\mu$m)')
                
    elif n_parameters_pop1 == 3 and n_parameters_pop2 ==2:
        scaling_factor, p, a_min, a_max, p2, a_min2, a_max2, f_B, f_C, f_D, f_B2, f_C2  = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))    
        porosity = np.ones((3)) - f_B - f_C -f_D
        porosity2 = np.ones((3)) - f_B2 - f_C2
        mcmc_values = np.array([scaling_factor, p, a_min, a_max, p2, a_min2, a_max2, porosity, f_B, f_C, f_D, porosity2, f_B2, f_C2])

        best_fit_para = [scaling_factor[0], p[0], a_min[0], a_max[0], p2[0], a_min2[0], a_max2[0],   f_B[0], f_C[0], f_D[0],   f_B2[0], f_C2[0]]  
        
        model_bestfit = generate_model_spectrum_mcmc_two_pop(best_fit_para) 
        n_freedom = np.count_nonzero(~np.isnan(unc))-len( best_fit_para)
        if np.count_nonzero(model_bestfit) == 0:
            chi2_reduced =  np.inf
        else:
            chi2_reduced = chi2(data, unc, model_bestfit, lnlike = False)/n_freedom

        labels = [r'$f$', r'$p$', r'$a_{min}$', r'$a_{max}$', r'$p2$', r'$a_{min2}$', r'$a_{max2}$', 
                 r'v_{0}'.format(comp1_name), r'v_{0}'.format(comp2_name), r'v_{0}'.format(comp3_name), 
                 r'v_{0}'.format(complarge_name), r'v_{0}'.format(complarge2_name)   ]

        stats = (f'$\chi^{2}_{{v}}$ = {chi2_reduced:.2f}\n'
                f'$f$ = {scaling_factor[0]:.3f} \n'
                f'$a_{{min}}$ = {a_min[0]:.2f} ($\mu$m)\n'
                f'$a_{{max}}$ = {a_max[0]:.2f} ($\mu$m)\n'
                f'$p$ = {p[0]:.2f}\n'
                f'porosity = {porosity[0]:.2f}\n'
                f'{comp1_name} ({ice_T_amp:.0f}K) = {f_B[0]:.2f}\n'
                f'{comp2_name} ({ice_T:.0f}K) = {f_C[0]:.2f}\n'
                f'{comp3_name}= {f_D[0]:.2f}\n'
                f'$p2$ = {p2[0]:.2f}\n'
                f'porosity2 = {porosity2[0]:.2f}\n'
                f'{complarge_name} ({ice_T:.0f}K) = {f_B2[0]:.2f}\n'
                f'{complarge2_name}= {f_C2[0]:.2f}\n'
                f'$a_{{min2}}$ = {a_min2[0]:.2f} ($\mu$m)\n'
                f'$a_{{max2}}$ = {a_max2[0]:.2f} ($\mu$m)')

    elif n_parameters_pop1 == 4 and n_parameters_pop2 ==2:
        scaling_factor, p, a_min, a_max, p2, a_min2, a_max2, f_B, f_C, f_D, f_E,  f_B2, f_C2  = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))    
        porosity = np.ones((3)) - f_B - f_C - f_D - f_E
        porosity2 = np.ones((3)) - f_B2 - f_C2 # the errorbar may not correct!
        mcmc_values = np.array([scaling_factor, p, a_min, a_max, p2, a_min2, a_max2, f_B, f_C, f_D, f_E,  f_B2, f_C2])

        best_fit_para = [scaling_factor[0], p[0], a_min[0], a_max[0], p2[0], a_min2[0], a_max2[0],   f_B[0], f_C[0], f_D[0], f_E[0],   f_B2[0], f_C2[0]]  
 
        model_bestfit = generate_model_spectrum_mcmc_two_pop(best_fit_para) 
        n_freedom = np.count_nonzero(~np.isnan(unc))-len( best_fit_para)
        if np.count_nonzero(model_bestfit) == 0:
            chi2_reduced =  np.inf
        else:
            chi2_reduced = chi2(data, unc, model_bestfit, lnlike = False)/n_freedom
            
        labels = [r'$f$', r'$p$', r'$a_{min}$', r'$a_{max}$', r'$p2$', r'$a_{min2}$', r'$a_{max2}$', 
                 r'v_{0}'.format(comp1_name), r'v_{0}'.format(comp2_name), r'v_{0}'.format(comp3_name), r'v_{0}'.format(comp4_name), 
                 r'v_{0}'.format(complarge_name), r'v_{0}'.format(complarge2_name)   ]

        stats = (f'$\chi^{2}_{{v}}$ = {chi2_reduced:.2f}\n'
                f'$f$ = {scaling_factor[0]:.3f} \n'
                f'$a_{{min}}$ = {a_min[0]:.2f} ($\mu$m)\n'
                f'$a_{{max}}$ = {a_max[0]:.2f} ($\mu$m)\n'
                f'$p$ = {p[0]:.2f}\n'
                f'porosity = {porosity[0]:.2f}\n'
                f'{comp1_name} ({ice_T_amp:.0f}K) = {f_B[0]:.2f}\n'
                f'{comp2_name} ({ice_T:.0f}K) = {f_C[0]:.2f}\n'
                f'{comp3_name}= {f_D[0]:.2f}\n'
                f'{comp4_name} = {f_E[0]:.2f}\n'
                f'$p2$ = {p2[0]:.2f}\n'
                f'porosity2 = {porosity2[0]:.2f}\n'
                f'{complarge_name} ({ice_T:.0f}K) = {f_B2[0]:.2f}\n'
                f'{complarge2_name}= {f_C2[0]:.2f}\n'
                f'$a_{{min2}}$ = {a_min2[0]:.2f} ($\mu$m)\n'
                f'$a_{{max2}}$ = {a_max2[0]:.2f} ($\mu$m)')

    elif n_parameters_pop1 == 5 and n_parameters_pop2 ==2:
        scaling_factor, p, a_min, a_max, p2, a_min2, a_max2, f_B, f_C, f_D, f_E, f_F,  f_B2, f_C2  = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))    
        porosity = np.ones((3)) - f_B - f_C - f_D - f_E - f_F
        porosity2 = np.ones((3)) - f_B2 - f_C2
        mcmc_values = np.array([scaling_factor, p, a_min, a_max, p2, a_min2, a_max2, porosity, f_B, f_C, f_D, f_E, f_F, porosity2, f_B2, f_C2])

        best_fit_para = [scaling_factor[0], p[0], a_min[0], a_max[0], p2[0], a_min2[0], a_max2[0],   f_B[0], f_C[0], f_D[0], f_E[0], f_F[0],   f_B2[0], f_C2[0]]  

        model_bestfit = generate_model_spectrum_mcmc_two_pop(best_fit_para) 
        n_freedom = np.count_nonzero(~np.isnan(unc))-len( best_fit_para)
        if np.count_nonzero(model_bestfit) == 0:
            chi2_reduced =  np.inf
        else:
            chi2_reduced = chi2(data, unc, model_bestfit, lnlike = False)/n_freedom

        labels = [r'$f$', r'$p$', r'$a_{min}$', r'$a_{max}$', r'$p2$', r'$a_{min2}$', r'$a_{max2}$', 
                 r'v_{0}'.format(comp1_name), r'v_{0}'.format(comp2_name), r'v_{0}'.format(comp3_name), r'v_{0}'.format(comp4_name), r'v_{0}'.format(comp5_name), 
                 r'v_{0}'.format(complarge_name), r'v_{0}'.format(complarge2_name)   ]

        stats = (f'$\chi^{2}_{{v}}$ = {chi2_reduced:.2f}\n'
                f'$f$ = {scaling_factor[0]:.3f} \n'
                f'$a_{{min}}$ = {a_min[0]:.2f} ($\mu$m)\n'
                f'$a_{{max}}$ = {a_max[0]:.2f} ($\mu$m)\n'
                f'$p$ = {p[0]:.2f}\n'
                f'porosity = {porosity[0]:.2f}\n'
                f'{comp1_name} ({ice_T_amp:.0f}K) = {f_B[0]:.2f}\n'
                f'{comp2_name} ({ice_T:.0f}K) = {f_C[0]:.2f}\n'
                f'{comp3_name}= {f_D[0]:.2f}\n'
                f'{comp4_name} = {f_E[0]:.2f}\n'
                f'{comp5_name} = {f_F[0]:.2f}\n'
                f'$p2$ = {p2[0]:.2f}\n'
                f'porosity2 = {porosity2[0]:.2f}\n'
                f'{complarge_name} ({ice_T:.0f}K) = {f_B2[0]:.2f}\n'
                f'{complarge2_name}= {f_C2[0]:.2f}\n'
                f'$a_{{min2}}$ = {a_min2[0]:.2f} ($\mu$m)\n'
                f'$a_{{max2}}$ = {a_max2[0]:.2f} ($\mu$m)')

    elif n_parameters_pop1 == 6 and n_parameters_pop2 ==2:
        scaling_factor, p, a_min, a_max, p2, a_min2, a_max2, f_B, f_C, f_D, f_E, f_F, f_G,  f_B2, f_C2  = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))    
        porosity = np.ones((3)) - f_B - f_C - f_D - f_E - f_F - f_G
        porosity2 = np.ones((3)) - f_B2 - f_C2
        mcmc_values = np.array([scaling_factor, p, a_min, a_max, p2, a_min2, a_max2, porosity, f_B, f_C, f_D, f_E, f_F, f_G, porosity2, f_B2, f_C2])

        best_fit_para = [scaling_factor[0], p[0], a_min[0], a_max[0], p2[0], a_min2[0], a_max2[0],   f_B[0], f_C[0], f_D[0], f_E[0], f_F[0], f_G[0],  f_B2[0], f_C2[0]]  
  
        model_bestfit = generate_model_spectrum_mcmc_two_pop(best_fit_para) 
        n_freedom = np.count_nonzero(~np.isnan(unc))-len( best_fit_para)
        if np.count_nonzero(model_bestfit) == 0:
            chi2_reduced =  np.inf
        else:
            chi2_reduced = chi2(data, unc, model_bestfit, lnlike = False)/n_freedom

        labels = [r'$f$', r'$p$', r'$a_{min}$', r'$a_{max}$', r'$p2$', r'$a_{min2}$', r'$a_{max2}$', 
                 r'v_{0}'.format(comp1_name), r'v_{0}'.format(comp2_name), r'v_{0}'.format(comp3_name), r'v_{0}'.format(comp4_name), r'v_{0}'.format(comp5_name), r'v_{0}'.format(comp6_name), 
                 r'v_{0}'.format(complarge_name), r'v_{0}'.format(complarge2_name)   ]


        stats = (f'$\chi^{2}_{{v}}$ = {chi2_reduced:.2f}\n'
                f'$f$ = {scaling_factor[0]:.3f} \n'
                f'$a_{{min}}$ = {a_min[0]:.2f} ($\mu$m)\n'
                f'$a_{{max}}$ = {a_max[0]:.2f} ($\mu$m)\n'
                f'$p$ = {p[0]:.2f}\n'
                f'porosity = {porosity[0]:.2f}\n'
                f'{comp1_name} ({ice_T_amp:.0f}K) = {f_B[0]:.2f}\n'
                f'{comp2_name} ({ice_T:.0f}K) = {f_C[0]:.2f}\n'
                f'{comp3_name}= {f_D[0]:.2f}\n'
                f'{comp4_name} = {f_E[0]:.2f}\n'
                f'{comp5_name} = {f_F[0]:.2f}\n'
                f'{comp6_name} = {f_G[0]:.2f}\n'
                f'$p2$ = {p2[0]:.2f}\n'
                f'porosity2 = {porosity2[0]:.2f}\n'
                f'{complarge_name} ({ice_T:.0f}K) = {f_B2[0]:.2f}\n'
                f'{complarge2_name}= {f_C2[0]:.2f}\n'
                f'$a_{{min2}}$ = {a_min2[0]:.2f} ($\mu$m)\n'
                f'$a_{{max2}}$ = {a_max2[0]:.2f} ($\mu$m)')


    color ='#EA8379'
    # plotname = '/new_mie_SB_{0}_{1}_{2}_{3}_{4}'.format(round(a_min[0],2), round(a_max[0],2), round(p[0],3), round(chi2_reduced,3), spec_region) 
    plotname = '/new_mie_SB_{0}_{1}_{2}_{3}_{4}_nziter_{5}_log_num{6}'.format(round(a_min[0],2), round(a_max[0],2), round(p[0],3), round(chi2_reduced,3), spec_region, nz_iter , num_integ) 
    plot_spectrum_errorbar(disk_spec[:,0], disk_spec[:,1], model_bestfit , z_wave, plotname, savepath, color, stats)
    plot_spectrum_errorbar(disk_spec[:,0], disk_spec[:,1], model_bestfit , z_wave, '/check_output.pdf', root, color, stats)

    print('#########  making best-fit model result  #########')
    return  model_bestfit, mcmc_values, chi2_reduced, labels



#####################################################



timeStamps = [time.time()]

################################################################
#######                       Path                     #########
################################################################
root = '/Users/sxie/repos/DDRM'
optical_constant_path  = root + '/optical_constants/'
disk_data_path = root + '/example/disk_spectrum_data'
output_model_path = root + '/example/model_spectrum' 
output_fitting_path = root + '/example/fitting_plots'


#################################
savepath = output_fitting_path
#################################
################################################################
#######                     input data                 #########
################################################################
res, header = fits.getdata(disk_data_path + '/residual_single_frame_sub_RDI_HD181327_IFU_align.fits', header=True)
nz = res.shape[0]
z_wave = np.arange(header['CRVAL3'], header['CRVAL3']+ (nz)*header['CDELT3'], header['CDELT3'], )  # NIRSpec wavelength coverage

disk_spectrum = fits.getdata(disk_data_path + '/HD181327_IFU_align_corrected_disk_spectrum_mid.fits')[:,1:3]

spec_region = 'inner'

scaling_factor = 1/disk_spectrum[380, 0]     # normalized to NIRSpec IFU channel #380, which is 2.5 micron
disk_spec = disk_spectrum*scaling_factor

disk_spec = disk_spec[115:-21]  #  NIRSpec IFU 1.1-7.2 microns.
z_wave = z_wave[115:-21]
nz = z_wave.shape[0]

################################################################
#######               model parameters                 #########
################################################################
g= [0.3]
angles = [90]
f = hgg_phase_function(math.radians(angles[0]), g, rayleigh_pol = False)
print('####  HG funtion f:', f)

nz_iter = 300       # wavelength sampling 
num_integ = 500     # sampling in calculating the model reflectance at a given wavelength; 500 is sufficient
wave_inter = np.linspace(z_wave[0], z_wave[-1], nz_iter, )
n_parameters_pop1 = 4
n_parameters_pop2 = 2   # if no zero, then using two dust population in the reflectance model

ice_T_amp= 50   # water ice temperature
ice_T= 50      # water ice temperature

# mcmc parameters
step =  10                      # how many steps are expected for MCMC to runx0
trunc = 200                                        # A step number you'd like to truncate at (aim: get rid of the burrning stage)


initial_guess = [complex(2, 1)] # most refractive indices range between 1 and 3
################################################################
#######               optical constants                #########
################################################################
# Amorphous_T = [15,25,40,50,60,80,100,120]
# Crystalline_T = [20,30,40,50,60,70,80,90,100,110,120,130,140,150]
ice_type='Amorphous'
comp1_filename = ['water_ice_{0}_{1}_Mastrapa_nk.txt'.format(ice_type, ice_T_amp,), ]
comp1_name = r'H$_2$O (Amorp)'

ice_type='Crystalline' 
comp2_filename = ['water_ice_{0}_{1}_Mastrapa_nk.txt'.format(ice_type, ice_T,), ]
comp2_name = r'H$_2$O (Cryst)'

comp3_filename = ['fes-Henning1996.lnk', ]
comp3_name = r'FeS'

comp4_filename = ['Olivine_modif.dat', ]
comp4_name = r'olivine'

comp5_filename = ['sic-Draine1993.lnk', ]
comp5_name = 'SiC'

comp6_filename = ['astrosil-Draine2003.lnk', ]
comp6_name = r'astrosil'

# second dust population
complarge_filename = ['water_ice_{0}_{1}_Mastrapa_nk.txt'.format(ice_type, ice_T,), ] 
complarge_name = r'H$_2$O (Cryst)'

complarge2_filename = ['Olivine_modif.dat' ]
complarge2_name = r'olivine'
# complarge2_filename = ['c-p-Preibisch1993.lnk' ]
# complarge2_filename = ['nh3-m-Martonchik1983.txt', ]


################################################################
#######     initial inputs; two dust population        #########
################################################################
"""
two dust population
model parameters:
[0, 1,   2,    3,    4,    5,       6,     7,   8,   ... ...  ,  -2,  -1 ]
[f, p, a_min, a_max, p2, a_min2, a_max2,  f_B, f_C,  ... ...  , f_B2, f_C2]
"""
if n_parameters_pop1 == 2 and n_parameters_pop2 ==2:
    x_all = [1, -3.5, 1, 100,  -3.5, 2, 100,     0.5, 0.15,     0.2, 0.2]
    bounds = np.array([[0, -4.5, 0, 20,  -4.5, 0, 20,   0, 0,   0, 0],[1, -2, 20, 1000, -2, 20, 1000,  1, 1,   1, 1]])

    comp1_file = optical_constant_path + comp1_filename[0]
    comp2_file = optical_constant_path + comp2_filename[0]

    complarge_file = optical_constant_path + complarge_filename[0] 
    complarge_file2 = optical_constant_path + complarge2_filename[0]

    w, n_B = inter_n_i(comp1_file, wave_inter,)
    w, n_C = inter_n_i(comp2_file, wave_inter,)

    w, n_B2 = inter_n_i(complarge_file, wave_inter,) 
    w, n_C2 = inter_n_i(complarge_file2, wave_inter,) 
 
    fitted_model_spec = np.zeros((nz,2))
    fitted_model_spec[:,0] = z_wave

    n_array = np.zeros((n_parameters_pop1+1, nz_iter), dtype=complex)
    n_array[0,:] = np.ones((nz_iter), dtype=complex)
    n_array[1,:] = n_B
    n_array[2,:] = n_C

    n_array2 = np.zeros((n_parameters_pop2+1, nz_iter), dtype=complex)
    n_array2[0,:] = np.ones((nz_iter), dtype=complex)
    n_array2[1,:] = n_B2
    n_array2[2,:] = n_C2

elif n_parameters_pop1 == 3 and n_parameters_pop2 ==2:
    x_all = [1, -3.5, 1, 100,  -3.5, 2, 100,     0.5, 0.15, 0.1,     0.2, 0.2]
    bounds = np.array([[0, -4.5, 0, 20,  -4.5, 0, 20,   0, 0, 0,   0, 0],[1, -2, 20, 1000, -2, 20, 1000,  1, 1, 1,   1, 1]])

    comp1_file = optical_constant_path + comp1_filename[0]
    comp2_file = optical_constant_path + comp2_filename[0]
    comp3_file = optical_constant_path + comp3_filename[0]

    complarge_file = optical_constant_path + complarge_filename[0] 
    complarge_file2 = optical_constant_path + complarge2_filename[0]

    w, n_B = inter_n_i(comp1_file, wave_inter,)
    w, n_C = inter_n_i(comp2_file, wave_inter,)
    w, n_D = inter_n_i(comp3_file, wave_inter,)

    w, n_B2 = inter_n_i(complarge_file, wave_inter,) 
    w, n_C2 = inter_n_i(complarge_file2, wave_inter,) 
 
    fitted_model_spec = np.zeros((nz,2))
    fitted_model_spec[:,0] = z_wave

    n_array = np.zeros((n_parameters_pop1+1, nz_iter), dtype=complex)
    n_array[0,:] = np.ones((nz_iter), dtype=complex)
    n_array[1,:] = n_B
    n_array[2,:] = n_C
    n_array[3,:] = n_D

    n_array2 = np.zeros((n_parameters_pop2+1, nz_iter), dtype=complex)
    n_array2[0,:] = np.ones((nz_iter), dtype=complex)
    n_array2[1,:] = n_B2
    n_array2[2,:] = n_C2

elif n_parameters_pop1 == 4 and n_parameters_pop2 ==2:
    x_all = [2, -3.5, 1.2, 400,  -3.6, 1.2, 400,     0.001, 0.001, 0.35, 0.05,      0.8, 0.05] # out 
    bounds = np.array([[0, -5, 0.02, 20,  -5, 0.01, 20,   0, 0, 0, 0,   0, 0],[20, -2, 20, 1000, -2, 20, 1000,  1, 1, 1, 1,   1, 1]])

    comp1_file = optical_constant_path + comp1_filename[0]
    comp2_file = optical_constant_path + comp2_filename[0]
    comp3_file = optical_constant_path + comp3_filename[0]
    comp4_file = optical_constant_path + comp4_filename[0]

    complarge_file = optical_constant_path + complarge_filename[0] 
    complarge_file2 = optical_constant_path + complarge2_filename[0]

    w, n_B = inter_n_i(comp1_file, wave_inter,)
    w, n_C = inter_n_i(comp2_file, wave_inter,)
    w, n_D = inter_n_i(comp3_file, wave_inter,)
    w, n_E = inter_n_i(comp4_file, wave_inter,)

    w, n_B2 = inter_n_i(complarge_file, wave_inter,) 
    w, n_C2 = inter_n_i(complarge_file2, wave_inter,) 
 
    fitted_model_spec = np.zeros((nz,2))
    fitted_model_spec[:,0] = z_wave

    n_array = np.zeros((n_parameters_pop1+1, nz_iter), dtype=complex)
    n_array[0,:] = np.ones((nz_iter), dtype=complex)
    n_array[1,:] = n_B
    n_array[2,:] = n_C
    n_array[3,:] = n_D
    n_array[4,:] = n_E

    n_array2 = np.zeros((n_parameters_pop2+1, nz_iter), dtype=complex)
    n_array2[0,:] = np.ones((nz_iter), dtype=complex)
    n_array2[1,:] = n_B2
    n_array2[2,:] = n_C2

elif n_parameters_pop1 == 5 and n_parameters_pop2 ==2:
    x_all = [1, -3.5, 1, 100,  -3.5, 2, 100,     0.5, 0.15, 0.1, 0.1, 0.1,      0.2, 0.2]
    bounds = np.array([[0, -4.5, 0, 20,  -4.5, 0, 20,   0, 0, 0, 0, 0,   0, 0],[1, -2, 20, 1000, -2, 20, 1000,  1, 1, 1, 1, 1,   1, 1]])

    comp1_file = optical_constant_path + comp1_filename[0]
    comp2_file = optical_constant_path + comp2_filename[0]
    comp3_file = optical_constant_path + comp3_filename[0]
    comp4_file = optical_constant_path + comp4_filename[0]
    comp5_file = optical_constant_path + comp5_filename[0]

    complarge_file = optical_constant_path + complarge_filename[0] 
    complarge_file2 = optical_constant_path + complarge2_filename[0]

    w, n_B = inter_n_i(comp1_file, wave_inter,)
    w, n_C = inter_n_i(comp2_file, wave_inter,)
    w, n_D = inter_n_i(comp3_file, wave_inter,)
    w, n_E = inter_n_i(comp4_file, wave_inter,)
    w, n_F = inter_n_i(comp5_file, wave_inter,)

    w, n_B2 = inter_n_i(complarge_file, wave_inter,) 
    w, n_C2 = inter_n_i(complarge_file2, wave_inter,) 
 
    fitted_model_spec = np.zeros((nz,2))
    fitted_model_spec[:,0] = z_wave

    n_array = np.zeros((n_parameters_pop1+1, nz_iter), dtype=complex)
    n_array[0,:] = np.ones((nz_iter), dtype=complex)
    n_array[1,:] = n_B
    n_array[2,:] = n_C
    n_array[3,:] = n_D
    n_array[4,:] = n_E
    n_array[5,:] = n_F

    n_array2 = np.zeros((n_parameters_pop2+1, nz_iter), dtype=complex)
    n_array2[0,:] = np.ones((nz_iter), dtype=complex)
    n_array2[1,:] = n_B2
    n_array2[2,:] = n_C2

elif n_parameters_pop1 == 6 and n_parameters_pop2 ==2:
    x_all = [1, -3.5, 1, 100,  -3.5, 2, 100,     0.5, 0.15, 0.1, 0.1, 0.1, 0.1,     0.2, 0.2]
    bounds = np.array([[0, -4.5, 0, 20,  -4.5, 0, 20,   0, 0, 0, 0, 0, 0,   0, 0],[1, -2, 20, 1000, -2, 20, 1000,  1, 1, 1, 1, 1, 1,   1, 1]])

    comp1_file = optical_constant_path + comp1_filename[0]
    comp2_file = optical_constant_path + comp2_filename[0]
    comp3_file = optical_constant_path + comp3_filename[0]
    comp4_file = optical_constant_path + comp4_filename[0]
    comp5_file = optical_constant_path + comp5_filename[0]
    comp6_file = optical_constant_path + comp6_filename[0]

    complarge_file = optical_constant_path + complarge_filename[0] 
    complarge_file2 = optical_constant_path + complarge2_filename[0]

    w, n_B = inter_n_i(comp1_file, wave_inter,)
    w, n_C = inter_n_i(comp2_file, wave_inter,)
    w, n_D = inter_n_i(comp3_file, wave_inter,)
    w, n_E = inter_n_i(comp4_file, wave_inter,)
    w, n_F = inter_n_i(comp5_file, wave_inter,)
    w, n_G = inter_n_i(comp6_file, wave_inter,)

    w, n_B2 = inter_n_i(complarge_file, wave_inter,) 
    w, n_C2 = inter_n_i(complarge_file2, wave_inter,) 
 
    fitted_model_spec = np.zeros((nz,2))
    fitted_model_spec[:,0] = z_wave

    n_array = np.zeros((n_parameters_pop1+1, nz_iter), dtype=complex)
    n_array[0,:] = np.ones((nz_iter), dtype=complex)
    n_array[1,:] = n_B
    n_array[2,:] = n_C
    n_array[3,:] = n_D
    n_array[4,:] = n_E
    n_array[5,:] = n_F
    n_array[6,:] = n_G

    n_array2 = np.zeros((n_parameters_pop2+1, nz_iter), dtype=complex)
    n_array2[0,:] = np.ones((nz_iter), dtype=complex)
    n_array2[1,:] = n_B2
    n_array2[2,:] = n_C2



#################################################################################
#######       Start spec fitting;     mcmc;    two dust population      #########
#################################################################################

n_dim = len(x_all)    # number of variables
n_walkers = int(4*n_dim)              # an even number (>= 2n_dim)
# n_walkers = int(10*n_dim)              # an even number (>= 2n_dim)

x = z_wave
data = disk_spec[:,0]
unc = disk_spec[:,1]


savepath = root + "/mcmc_result/{0}_nzinter_{1}_f_lognum_{2}_T_{3}/".format(spec_region, nz_iter, num_integ, ice_T)
if os.path.exists(savepath):
    filename = savepath+ "state_HD181327_dust_reflectance_Model_region_{0}_inter_{1}_lognum_{2}_T_{3}.h5".format(spec_region, nz_iter, num_integ, ice_T)
else:
    os.makedirs(savepath)
    filename = savepath+ "state_HD181327_dust_reflectance_Model_region_{0}_inter_{1}_lognum_{2}_T_{3}.h5".format(spec_region, nz_iter, num_integ, ice_T)


print('######## var_values_init:',  x_all) 
with Pool() as pool:
    start = time.time()

    if not os.path.exists(filename):  #initial run, no backend file existed
        backend = emcee.backends.HDFBackend(filename)   # the backend file is used to store the status
        sampler = emcee.EnsembleSampler(nwalkers = n_walkers, ndim = n_dim, log_prob_fn=lnpost_mcmc, pool = pool, backend=backend)

        # init_range = bounds
        var_values_init = x_all
        # values_ball2 = np.random.uniform(init_range[0], init_range[1], size = [n_walkers, n_dim]) #- init_range[0] + var_values_init
        # values_ball = [var_values_init + 1e-1*np.random.randn(n_dim) for i in range(n_walkers)] 
        n_counts=0
        values_ball =[]
        for i in range(n_walkers*1000):
            # tmp = var_values_init +  2e-1*np.random.randn(n_dim) 
            # tmp = np.random.uniform(bounds[0], bounds[1], size = [n_dim])
            tmp = np.array(var_values_init) + 0.5* np.array(var_values_init)*np.random.randn(n_dim) 

            if  inspect_parameter(tmp) == -np.inf: 
                pass
            else:
                values_ball.append(tmp)
                n_counts += 1
                if n_counts == n_walkers:
                    break

        check_bounds = np.zeros((len(values_ball)))
        for i in range(len(values_ball)):
            if  inspect_parameter(tmp) == -np.inf:
                check_bounds[i] = 1 
        if np.count_nonzero(check_bounds) > 0:
            print('Initial inputs outside bounds')
            print('Stoping MCMC')
        elif np.array(values_ball).shape[0] != n_walkers:
            print('Insufficient inputs')
            print('Stoping MCMC')
        else:
            print('Initial inputs are wiithin the valid range')
            print('Starting MCMC')
            # print(values_ball)
            fits.writeto(savepath + 'HD181327_dust_model_input_parameters_{0}.fits'.format(spec_region), np.array(values_ball), overwrite=True)
            sampler.run_mcmc(values_ball, step, progress=True)


    else:    #load the data directly from a backend file, this is used when you want to pick up from a previous MCMC run
        backend = emcee.backends.HDFBackend(filename)   # the backend file is used to store the status
        sampler = emcee.EnsembleSampler(nwalkers = n_walkers, ndim = n_dim, log_prob_fn=lnpost_mcmc, pool = pool, backend=backend)
        sampler.run_mcmc(None, nsteps = step, progress=True)

    end = time.time()
    serial_time = end - start

    print("1 nodes * {0} cores {1} steps with multiprocess took {2:.1f} seconds".format(cpu_count(), step, serial_time))


samples = sampler.chain[:, trunc:, :].reshape((-1, n_dim))
model_bestfit, mcmc_values, chi2_reduced, labels = output_best_fit_mcmc_result_two_pop(samples) 

fits.writeto(savepath + 'HD181327_dust_model_best_fit_{0}.fits'.format(spec_region), model_bestfit, overwrite=True)
fits.writeto(savepath + 'HD181327_mcmc_values_{0}.fits'.format(spec_region), mcmc_values, overwrite=True)

output_cornor_mcmc_result(samples, mcmc_values, labels)

print('Reduced chi2 is: ', chi2_reduced)
# # ########################

timeStamps.append(time.time())
totalTime = timeStamps[-1]-timeStamps[0]
print('-- Total Processing time: ', totalTime, ' s')
print('')

