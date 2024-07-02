#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
import corner

# %% FUNCTIONS



def plot_spectrum_errorbar(input_1D_spec, input_1D_spec_error, model, z_wave, plotname, savepath, color, stats ):
    pdfname = savepath + '{0}.pdf'.format(plotname)
    fig = plt.figure(figsize=(10,7.2), dpi=300)
    fig.subplots_adjust(hspace=0.0, wspace=0.0001, bottom=0.08, top=0.96, left=0.1, right=0.96)
    gs = gridspec.GridSpec(2, 1,height_ratios=[2,1])
    plt.rcParams['xtick.direction'] = 'in' 
    plt.rcParams['ytick.direction'] = 'in' 
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True

    nz = input_1D_spec.shape[0]
    topplot = plt.subplot(gs[0], )
    plt.errorbar(z_wave, input_1D_spec, input_1D_spec_error, 0,  color = color, label='disk spectrum', fmt='o', markersize=4, alpha=0.3, ls='none', zorder=6)
    plt.plot(z_wave, model, lw=1.5, alpha=1, color = 'gray', ls='-', label='model spectrum', zorder=10)
    # plt.plot(z_wave, model2+0.5, lw=1.5, alpha=1,  ls='-', label='model spectrum (halo dust) + 0.5', zorder=10)

    # plt.title(legend)
    ylim_min = 0.5   
    ylim_max = 2.25
    # ylim_min = 0.5   
    # ylim_max = 1.5
    plt.ylim([ylim_min, ylim_max])
    xlim_min = 0.5    
    xlim_max = 5.5
    plt.xlim([xlim_min, xlim_max])
    plt.tick_params(axis='both', which='major', labelsize=16)
    legend = plt.legend(loc='upper right', fontsize=15,frameon=True, )

    plt.ylabel(r'$F_{\rm{disk}}$/$F_{\rm{star}}$ (normalized)',  fontsize=16)

    bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
    # plt.text(5.2, 0.4, stats, fontsize=12, bbox=bbox,  horizontalalignment='right')
    plt.text(2.1, 1, stats, fontsize=12, bbox=bbox,  horizontalalignment='right')

    topplot.tick_params(axis='both', which='major', labelsize=16, length=8, width=1.6, pad=10)
    for axis in ['top','bottom','left','right']:
        topplot.spines[axis].set_linewidth(1.6)
    plt.setp(topplot.get_xticklabels(), visible=False)

    bottomplot = plt.subplot(gs[1], sharex = topplot )
    input_1D_spec_error[np.where(input_1D_spec_error <= 0)] = np.nan
    chi2_i = np.zeros((z_wave.shape[0]))
    for z in range(z_wave.shape[0]):
        chi2_i[z] = ((input_1D_spec[z]-model[z])/input_1D_spec_error[z])**2

    plt.plot(z_wave, chi2_i, lw=1.5, alpha=1, color = 'gray', ls='-', label=r'$\chi$$^{2}_{\lambda}$', zorder=10)
    xlim_min = 0.5    
    xlim_max = 5.5
    plt.xlim([xlim_min, xlim_max])
    plt.tick_params(axis='both', which='major', labelsize=16)

    plt.xlabel(r'Wavelength ($\rm \mu m$)', fontsize=16)
    plt.ylabel(r'$\chi$$^{2}_{\lambda}$',  fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=16)

    bottomplot.tick_params(axis='both', which='major', labelsize=16, length=8, width=1.6, pad=10)
    for axis in ['top','bottom','left','right']:
        bottomplot.spines[axis].set_linewidth(1.6)
    legend = plt.legend(loc='best', fontsize=15,frameon=True)
    fig.align_ylabels()
    plt.savefig(pdfname, transparent= True)
    plt.clf()



def find_wave_indice(input_data, short_wave, long_wave):
    short_wave_indice = np.nanargmin(abs(input_data[:,0]-short_wave), axis=0)
    long_wave_indice = np.nanargmin(abs(input_data[:,0]-long_wave), axis=0)
    return short_wave_indice, long_wave_indice 
    


def hgg_phase_function(phi,g, rayleigh_pol = False):
    #Inputs:
    # g - the g 
    # phi - the scattering angle in radians
    g = g[0]

    cos_phi = np.cos(phi)
    g2p1 = g**2 + 1
    gg = 2*g
    k = 1./(4*np.pi)*(1-g*g)
    if not rayleigh_pol:
        return k/(g2p1 - (gg*cos_phi))**1.5
    else: #HGG modified by Rayleigh scattering to generate polarized image, added by Bin Ren
        cos2phi = cos_phi**2
        return k/(g2p1 - (gg*cos_phi))**1.5 * (1 - cos2phi)/(1 + cos2phi)



def inter_n_i(input_n_file, wave_inter ,):
    binary_data = np.loadtxt(input_n_file, usecols=[0,1,2])
    n_array = np.zeros((binary_data[:,0].shape[0]), dtype=complex)
    for z in range(binary_data[:,0].shape[0]):
        n_array[z] = complex(binary_data[z,1], binary_data[z,2])

    refr_index_cube_n = interp1d(binary_data[:,0], binary_data[:,1] )
    refr_index_cube_k = interp1d(binary_data[:,0], binary_data[:,2] )
    nz = len(wave_inter)
    refr_index_comp_i = np.zeros((nz), dtype=complex)
    for z in range(len(wave_inter)):
        refr_index_comp_i[z] = complex(refr_index_cube_n(wave_inter)[z], refr_index_cube_k(wave_inter)[z])
    
    return wave_inter, refr_index_comp_i,  



# def output_ice_nk(csv_data, ice_type, T):
#     # ice_type == 'Amorphous' or 'Crystalline'
#     crop_data = csv_data[csv_data['ice_type'] == ice_type]
#     crop_data = crop_data[crop_data['T'] == T]
#     crop_data_nk = np.zeros((3,len(crop_data)))
#     crop_data_nk[0,:] = crop_data['lamda'] 
#     crop_data_nk[1,:] = crop_data['n'] 
#     crop_data_nk[2,:] = crop_data['k'] 
#     return crop_data_nk


# def create_water_ice_nk_txt(ice_path, optical_constant_path, ice_type, ice_T_amp):
    
#     csv_data = pd.read_csv(ice_path + '/water_ice_Mastrapa.csv',)
#     water_ice_nk = output_ice_nk(csv_data, ice_type=ice_type, T=ice_T_amp) 
#     ftxt = open(optical_constant_path +'/water_ice_{0}_{1}_Mastrapa_nk.txt'.format(ice_type, ice_T_amp,),"w") 
#     for z in range(water_ice_nk.shape[1]):
#         write_neff = str(water_ice_nk[0,z]) + '\t' + str(water_ice_nk[1,z]) + '\t' + str(water_ice_nk[2,z]) 
#         ftxt.write('%s\n' % write_neff)


def output_cornor_mcmc_result(samples, mcmc_values, labels, n_dim, savepath, spec_region):
    plt.figure(dpi=300)
    fig = corner.corner(samples, labels=labels, quantiles =[.16, .50, .84])         # corner plots with the quantiles  (-1sigma, median, +1sigma)
    axes = np.array(fig.axes).reshape((n_dim, n_dim))
    for yi in range(n_dim):
        for xi in range(n_dim):
            if xi == yi:
                ax = axes[yi, xi]
                title  = labels[xi] + '= {0}'.format(round(mcmc_values[xi, 0], 3)) + r'$^{{+{0}}}_{{-{1}}}$'.format(round(mcmc_values[xi, 1], 3), round(mcmc_values[xi, 2], 3))
                ax.set_title(title, color="k")

    plt.savefig(savepath + 'corner_{0}.pdf'.format(spec_region))
    plt.close()


    plt.figure(dpi=300)
    fig, axes = plt.subplots(n_dim, figsize=(7, 18), sharex=True)

    # samples = sampler.get_chain()
    # labels = ["m", "b", "log(f)"]
    labels.append('loglikelihood')
    for i in range(n_dim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        # ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")
    plt.savefig(savepath + 'time_series.pdf')
    plt.close()


