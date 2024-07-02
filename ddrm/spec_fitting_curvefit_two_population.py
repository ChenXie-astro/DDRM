#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import math
import time

from astropy.io import fits
from scipy.interpolate import interp1d
import miepython
from scipy.optimize import curve_fit
from mpmath import findroot
from utils import plot_spectrum_errorbar, hgg_phase_function, inter_n_i, 

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
    

def generate_model_spectrum_mcmc_two_pop(_, *x_all):
    """
    For 'curve_fit'
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
        n_bg_test =findroot(lambda n_bg: sum((vf[n]*(n_array[n,z]**2 - (n_bg)**2)/(n_array[n,z]**2 + 2*(n_bg)**2)) for n in np.arange(0, vf.shape[0])), initial_guess , solver='muller')
        n_bg_real = n_bg_test.real 
        n_bg_imag = n_bg_test.imag                    
        neff_brug[z] = complex(n_bg_real, -1* n_bg_imag)

        # I_nz_iter[z], err = quad(integrand, a_min, a_max, args=(neff_brug[z], wave_inter[z], f,  p), ) 
        # x_a = np.linspace(a_min, a_max, num=num_integ)
        x_a = np.logspace(np.log10(a_min), np.log10(a_max), num_integ)
        I_nz_iter[z] = np.trapz(integrand(x_a, neff_brug[z], wave_inter[z], f,  p), x_a)
        
        # large particles water ice
        n_bg_test2 =findroot(lambda n_bg: sum((vf2[n]*(n_array2[n,z]**2 - (n_bg)**2)/(n_array2[n,z]**2 + 2*(n_bg)**2)) for n in np.arange(0, vf2.shape[0])), initial_guess , solver='muller')
        n_bg_real2 = n_bg_test2.real 
        n_bg_imag2 = n_bg_test2.imag                    
        neff_brug2[z] = complex(n_bg_real2, -1* n_bg_imag2)

        # I_nz_iter2[z], err = quad(integrand, a_min2, a_max2, args=(neff_brug2[z], wave_inter[z], f,  p2), ) 
        # x_a2 = np.linspace(a_min2, a_max2, num=num_integ)
        x_a2 = np.logspace(np.log10(a_min2), np.log10(a_max2), num_integ)
        I_nz_iter2[z] = np.trapz(integrand(x_a2, neff_brug2[z], wave_inter[z], f,  p2), x_a2)

    f_I = interp1d(wave_inter, I_nz_iter+I_nz_iter2 )
    I = f_I(z_wave)* scaling_factor 

    return I


############## output plots  #########


def output_best_fit_result(x_all):
    # if n_parameters_pop1 == 2 and n_parameters_pop2 ==2:
    """
    model parameters:
    [0, 1,   2,    3,    4,    5,       6,     7,   8,   ... ...  ,  -2,  -1 ]
    [f, p, a_min, a_max, p2, a_min2, a_max2,  f_B, f_C,  ... ...  , f_B2, f_C2]
    """
    # dust population # 1
    if n_parameters_pop1 == 2 and n_parameters_pop2 ==2:
        scaling_factor = x_all[0] 
        p = x_all[1]
        a_min = x_all[2]
        a_max = x_all[3]
        # dust population # 2 only has two dust species # i.e., [p2, a_min2, a_max2, f_B2, f_C2] and porosity = 1-f_B2 - f_C2
        p2 = x_all[4] 
        a_min2 = x_all[5]
        a_max2 = x_all[6]  
        n_parameters = len(x_all)
        # number of components in the dust population #1 starts in n_parameters - 7
        # dust population #1
        N = n_parameters - 7 - n_parameters_pop2
        porosity = 1- sum(x_all[n+7] for n in np.arange(0, N))
        # dust population #2
        N2 = n_parameters_pop2
        porosity2 = 1- sum(x_all[n] for n in np.arange(-1*N2, 0))

        f_B = x_all[7]
        f_C = x_all[8]

        f_B2 = x_all[-2]
        f_C2 = x_all[-1]

        model_bestfit = generate_model_spectrum_mcmc_two_pop(z_wave, *x_all) 
        n_freedom = np.count_nonzero(~np.isnan(sigma))-len(x_all)
        if np.count_nonzero(model_bestfit) == 0:
            chi2_reduced =  np.inf
        else:
            chi2_reduced = chi2(disk_spec[:,0], disk_spec[:,1], model_bestfit, lnlike = False)/n_freedom
            
        stats = (f'$\chi^{2}_{{v}}$ = {chi2_reduced:.2f}\n'
                f'$f$ = {scaling_factor:.3f} \n'
                f'$a_{{min}}$ = {a_min:.2f} ($\mu$m)\n'
                f'$a_{{max}}$ = {a_max:.2f} ($\mu$m)\n'
                f'$p$ = {p:.2f}\n'
                f'porosity = {porosity:.2f}\n'
                f'{comp1_name} ({ice_T_amp:.0f}K) = {f_B:.2f}\n'
                f'{comp2_name} ({ice_T:.0f}K) = {f_C:.2f}\n'
                f'$p2$ = {p2:.2f}\n'
                f'porosity2 = {porosity2:.2f}\n'
                f'{complarge_name} ({ice_T:.0f}K) = {f_B2:.2f}\n'
                f'{complarge2_name}= {f_C2:.2f}\n'
                f'$a_{{min2}}$ = {a_min2:.2f} ($\mu$m)\n'
                f'$a_{{max2}}$ = {a_max2:.2f} ($\mu$m)')
                
    elif n_parameters_pop1 == 3 and n_parameters_pop2 ==2:
        scaling_factor = x_all[0] 
        p = x_all[1]
        a_min = x_all[2]
        a_max = x_all[3]
        # dust population # 2 only has two dust species # i.e., [p2, a_min2, a_max2, f_B2, f_C2] and porosity = 1-f_B2 - f_C2
        p2 = x_all[4] 
        a_min2 = x_all[5]
        a_max2 = x_all[6]  
        n_parameters = len(x_all)
        # number of components in the dust population #1 starts in n_parameters - 7
        # dust population #1
        N = n_parameters - 7 - n_parameters_pop2
        porosity = 1- sum(x_all[n+7] for n in np.arange(0, N))
        # dust population #2
        N2 = n_parameters_pop2
        porosity2 = 1- sum(x_all[n] for n in np.arange(-1*N2, 0))

        f_B = x_all[7]
        f_C = x_all[8]
        f_D = x_all[9]

        f_B2 = x_all[-2]
        f_C2 = x_all[-1]

        model_bestfit = generate_model_spectrum_mcmc_two_pop(z_wave, *x_all) 
        n_freedom = np.count_nonzero(~np.isnan(sigma))-len(x_all)
        if np.count_nonzero(model_bestfit) == 0:
            chi2_reduced =  np.inf
        else:
            chi2_reduced = chi2(disk_spec[:,0], disk_spec[:,1], model_bestfit, lnlike = False)/n_freedom
            
        stats = (f'$\chi^{2}_{{v}}$ = {chi2_reduced:.2f}\n'
                f'$f$ = {scaling_factor:.3f} \n'
                f'$a_{{min}}$ = {a_min:.2f} ($\mu$m)\n'
                f'$a_{{max}}$ = {a_max:.2f} ($\mu$m)\n'
                f'$p$ = {p:.2f}\n'
                f'porosity = {porosity:.2f}\n'
                f'{comp1_name} ({ice_T_amp:.0f}K) = {f_B:.2f}\n'
                f'{comp2_name} ({ice_T:.0f}K) = {f_C:.2f}\n'
                f'{comp3_name}= {f_D:.2f}\n'
                f'$p2$ = {p2:.2f}\n'
                f'porosity2 = {porosity2:.2f}\n'
                f'{complarge_name} ({ice_T:.0f}K) = {f_B2:.2f}\n'
                f'{complarge2_name}= {f_C2:.2f}\n'
                f'$a_{{min2}}$ = {a_min2:.2f} ($\mu$m)\n'
                f'$a_{{max2}}$ = {a_max2:.2f} ($\mu$m)')

    elif n_parameters_pop1 == 4 and n_parameters_pop2 ==2:
        scaling_factor = x_all[0] 
        p = x_all[1]
        a_min = x_all[2]
        a_max = x_all[3]
        # dust population # 2 only has two dust species # i.e., [p2, a_min2, a_max2, f_B2, f_C2] and porosity = 1-f_B2 - f_C2
        p2 = x_all[4] 
        a_min2 = x_all[5]
        a_max2 = x_all[6]  
        n_parameters = len(x_all)
        # number of components in the dust population #1 starts in n_parameters - 7
        # dust population #1
        N = n_parameters - 7 - n_parameters_pop2
        porosity = 1- sum(x_all[n+7] for n in np.arange(0, N))
        # dust population #2
        N2 = n_parameters_pop2
        porosity2 = 1- sum(x_all[n] for n in np.arange(-1*N2, 0))

        f_B = x_all[7]
        f_C = x_all[8]
        f_D = x_all[9]
        f_E = x_all[10]

        f_B2 = x_all[-2]
        f_C2 = x_all[-1]

        model_bestfit = generate_model_spectrum_mcmc_two_pop(z_wave, *x_all) 

        n_freedom = np.count_nonzero(~np.isnan(sigma))-len(x_all)
        print('n_freedom:',n_freedom)

        if np.count_nonzero(model_bestfit) == 0:
            chi2_reduced =  np.inf
        else:
            chi2_reduced = chi2(disk_spec[:,0], disk_spec[:,1], model_bestfit, lnlike = False)/n_freedom

        stats = (f'$\chi^{2}_{{v}}$ = {chi2_reduced:.2f}\n'
                f'$f$ = {scaling_factor:.3f} \n'
                f'$a_{{min}}$ = {a_min:.2f} ($\mu$m)\n'
                f'$a_{{max}}$ = {a_max:.2f} ($\mu$m)\n'
                f'$p$ = {p:.2f}\n'
                f'porosity = {porosity:.2f}\n'
                f'{comp1_name} ({ice_T_amp:.0f}K) = {f_B:.2f}\n'
                f'{comp2_name} ({ice_T:.0f}K) = {f_C:.2f}\n'
                f'{comp3_name}= {f_D:.2f}\n'
                f'{comp4_name} = {f_E:.2f}\n'
                f'$p2$ = {p2:.2f}\n'
                f'porosity2 = {porosity2:.2f}\n'
                f'{complarge_name} ({ice_T:.0f}K) = {f_B2:.2f}\n'
                f'{complarge2_name}= {f_C2:.2f}\n'
                f'$a_{{min2}}$ = {a_min2:.2f} ($\mu$m)\n'
                f'$a_{{max2}}$ = {a_max2:.2f} ($\mu$m)')

    elif n_parameters_pop1 == 5 and n_parameters_pop2 ==2:
        scaling_factor = x_all[0] 
        p = x_all[1]
        a_min = x_all[2]
        a_max = x_all[3]
        # dust population # 2 only has two dust species # i.e., [p2, a_min2, a_max2, f_B2, f_C2] and porosity = 1-f_B2 - f_C2
        p2 = x_all[4] 
        a_min2 = x_all[5]
        a_max2 = x_all[6]  
        n_parameters = len(x_all)
        # number of components in the dust population #1 starts in n_parameters - 7
        # dust population #1
        N = n_parameters - 7 - n_parameters_pop2
        porosity = 1- sum(x_all[n+7] for n in np.arange(0, N))
        # dust population #2
        N2 = n_parameters_pop2
        porosity2 = 1- sum(x_all[n] for n in np.arange(-1*N2, 0))

        f_B = x_all[7]
        f_C = x_all[8]
        f_D = x_all[9]
        f_E = x_all[10]
        f_F = x_all[11]

        f_B2 = x_all[-2]
        f_C2 = x_all[-1]

        model_bestfit = generate_model_spectrum_mcmc_two_pop(z_wave, *x_all)  
        n_freedom = np.count_nonzero(~np.isnan(sigma))-len(x_all)
        if np.count_nonzero(model_bestfit) == 0:
            chi2_reduced =  np.inf
        else:
            chi2_reduced = chi2(disk_spec[:,0], disk_spec[:,1], model_bestfit, lnlike = False)/n_freedom
            

        stats = (f'$\chi^{2}_{{v}}$ = {chi2_reduced:.2f}\n'
                f'$f$ = {scaling_factor:.3f} \n'
                f'$a_{{min}}$ = {a_min:.2f} ($\mu$m)\n'
                f'$a_{{max}}$ = {a_max:.2f} ($\mu$m)\n'
                f'$p$ = {p:.2f}\n'
                f'porosity = {porosity:.2f}\n'
                f'{comp1_name} ({ice_T_amp:.0f}K) = {f_B:.2f}\n'
                f'{comp2_name} ({ice_T:.0f}K) = {f_C:.2f}\n'
                f'{comp3_name}= {f_D:.2f}\n'
                f'{comp4_name} = {f_E:.2f}\n'
                f'{comp5_name} = {f_F:.2f}\n'
                f'$p2$ = {p2:.2f}\n'
                f'porosity2 = {porosity2:.2f}\n'
                f'{complarge_name} ({ice_T:.0f}K) = {f_B2:.2f}\n'
                f'{complarge2_name}= {f_C2:.2f}\n'
                f'$a_{{min2}}$ = {a_min2:.2f} ($\mu$m)\n'
                f'$a_{{max2}}$ = {a_max2:.2f} ($\mu$m)')

    elif n_parameters_pop1 == 6 and n_parameters_pop2 ==2:
        scaling_factor = x_all[0] 
        p = x_all[1]
        a_min = x_all[2]
        a_max = x_all[3]
        # dust population # 2 only has two dust species # i.e., [p2, a_min2, a_max2, f_B2, f_C2] and porosity = 1-f_B2 - f_C2
        p2 = x_all[4] 
        a_min2 = x_all[5]
        a_max2 = x_all[6]  
        n_parameters = len(x_all)
        # number of components in the dust population #1 starts in n_parameters - 7
        # dust population #1
        N = n_parameters - 7 - n_parameters_pop2
        porosity = 1- sum(x_all[n+7] for n in np.arange(0, N))
        # dust population #2
        N2 = n_parameters_pop2
        porosity2 = 1- sum(x_all[n] for n in np.arange(-1*N2, 0))

        f_B = x_all[7]
        f_C = x_all[8]
        f_D = x_all[9]
        f_E = x_all[10]
        f_F = x_all[11]
        f_G = x_all[11]

        f_B2 = x_all[-2]
        f_C2 = x_all[-1]

        model_bestfit = generate_model_spectrum_mcmc_two_pop(z_wave, *x_all) 
        n_freedom = np.count_nonzero(~np.isnan(sigma))-len(x_all)
        if np.count_nonzero(model_bestfit) == 0:
            chi2_reduced =  np.inf
        else:
            chi2_reduced = chi2(disk_spec[:,0], disk_spec[:,1], model_bestfit, lnlike = False)/n_freedom
            

        stats = (f'$\chi^{2}_{{v}}$ = {chi2_reduced:.2f}\n'
                f'$f$ = {scaling_factor:.3f} \n'
                f'$a_{{min}}$ = {a_min:.2f} ($\mu$m)\n'
                f'$a_{{max}}$ = {a_max:.2f} ($\mu$m)\n'
                f'$p$ = {p:.2f}\n'
                f'porosity = {porosity:.2f}\n'
                f'{comp1_name} ({ice_T_amp:.0f}K) = {f_B:.2f}\n'
                f'{comp2_name} ({ice_T:.0f}K) = {f_C:.2f}\n'
                f'{comp3_name}= {f_D:.2f}\n'
                f'{comp4_name} = {f_E:.2f}\n'
                f'{comp5_name} = {f_F:.2f}\n'
                f'{comp6_name} = {f_G:.2f}\n'
                f'$p2$ = {p2:.2f}\n'
                f'porosity2 = {porosity2:.2f}\n'
                f'{complarge_name} ({ice_T:.0f}K) = {f_B2:.2f}\n'
                f'{complarge2_name}= {f_C2:.2f}\n'
                f'$a_{{min2}}$ = {a_min2:.2f} ($\mu$m)\n'
                f'$a_{{max2}}$ = {a_max2:.2f} ($\mu$m)')


    color ='#EA8379'
    plotname = '/new_mie_SB_{0}_{1}_{2}_{3}_{4}_nziter_{5}_log_num{6}'.format(round(a_min,2), round(a_max,2), round(p,3), round(chi2_reduced,3), spec_region, nz_iter , num_integ) 

    plot_spectrum_errorbar(disk_spec[:,0], disk_spec[:,1], model_bestfit , z_wave, plotname, savepath, color, stats)
    plot_spectrum_errorbar(disk_spec[:,0], disk_spec[:,1], model_bestfit , z_wave, '/check_output.pdf', root, color, stats)

    print('#########  making best-fit model result  #########')
    return  model_bestfit, chi2_reduced 


#########################################




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

disk_spectrum = fits.getdata(disk_data_path + '/HD181327_IFU_align_corrected_disk_spectrum_inner.fits')[:,1:3]
spec_region = 'mid'

scaling_factor = 1/disk_spectrum[380, 0]  # normalized to NIRSpec IFU channel #380, which is 2.5 
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

# comp4_filename = ['pyr-mg100-Dorschner1995.lnk', ]
# comp4_name = r'MgFeSiO$_3$'
# comp4_filename = ['pyr-mg70-Dorschner1995.lnk', ]
# comp4_name = r'Mg$_{0.7}$Fe$_{0.3}$SiO$_3$'
# comp4_filename = ['pyr-mg40-Dorschner1995.lnk', ]
# comp4_name = r'Mg$_{0.4}$Fe$_{0.6}$SiO$_3$'

# second dust population
complarge_filename = ['water_ice_{0}_{1}_Mastrapa_nk.txt'.format(ice_type, ice_T,), ] 
complarge_name = r'H$_2$O (Cryst)'

complarge2_filename = ['Olivine_modif.dat' ]
complarge2_name = r'olivine'
# complarge2_filename = ['c-p-Preibisch1993.lnk' ]



################################################################
#######     initial inputs; single dust population     #########
################################################################
"""
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

    x_all = [2.8, -4, 1.4, 388,  -4, 2.6, 500,     0.001, 0.001, 0.05, 0.9,      0.35, 0.05]  # close to the best-fit
    bounds = np.array([[0, -5, 0.02, 20,  -5, 0.01, 20,   0, 0, 0, 0,   0, 0],[10, -2, 20, 1000, -2, 20, 1000,  1, 1, 1, 1,   1, 1]])

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


################################################################################
#######     Start spec fitting; curve_fit;  single dust population     #########
################################################################################

x = z_wave
y = disk_spec[:,0]
sigma = disk_spec[:,1]
expected = np.array(x_all)

popt_inital, pcov = curve_fit(generate_model_spectrum_mcmc_two_pop, x,y, expected, sigma=sigma, absolute_sigma=True, bounds=bounds,  maxfev=500)
popt, pcov = curve_fit(generate_model_spectrum_mcmc_two_pop, x, y, popt_inital, sigma=sigma, absolute_sigma=True, bounds=bounds,  maxfev=5000)
perr = np.sqrt(np.diag(pcov))
# # ########################
print('popt:', popt)
print('perr:', perr)
model_bestfit, chi2_reduced = output_best_fit_result(popt)
print('Reduced chi2 is: ', chi2_reduced)

fits.writeto(output_fitting_path + '/SB_normalized_scaling_{0}_{1}_{2}.fits'.format(round(popt[0],3), round(chi2_reduced,3), spec_region), model_bestfit, overwrite=True) 



timeStamps.append(time.time())
totalTime = timeStamps[-1]-timeStamps[0]
print('-- Total Processing time: ', totalTime, ' s')
print('')


