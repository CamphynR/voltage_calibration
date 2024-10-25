"""
This code contains helper functions for the code written in bias_scan_comparisons.ipynb
"""

import os
import json
import pickle
from contextlib import contextmanager

import uproot
import numpy as np
from scipy import interpolate
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import ROOT
from numba import jit
import numba as nb
import awkward as ak

@contextmanager # decorator allows you to define a function for a with statement without defining a  __enter__() and __exit__() method
def cwd(path):  # this specific function sets the working dorectory to path for all code in the with block
    oldpwd = os.getcwd()
    try:
        os.chdir(path)
    except OSError:
        os.mkdir(path)
        os.chdir(path)
        
    try:
        yield
    finally:
        os.chdir(oldpwd)


def read_config(config_path):
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    return config


def read_pickle(pickle_path):
    with open(pickle_path, "rb") as pickle_file:
        array = pickle.load(pickle_file)
    return array


def save_to_pickle(object, pickle_path):
    with open(pickle_path, "wb") as pickle_file:
        pickle.dump(object, pickle_file)
    return

def find_nr_subplots(n, threshold = 6):
    n0 = n
    while n > threshold:
        if n % 2 == 0:
            n = n/2
        elif n % 3 == 0:
            n = n/3
        elif n % 5 == 0:
            n = n/5
    return int(n0/n), int(n)



#Loads librno-g
ROOT.gSystem.Load('librno-g.so')
#Loads mattak
ROOT.gSystem.Load('libmattak.so')

def perform_fit(filepath : str, ref = 1.5) -> str:
    VC = ROOT.mattak.VoltageCalibration(filepath, ref) # -> takes a 'long' time to run, suffices to just read the file made by saveFitCoeffsInFile
    VC.saveFitCoeffsInFile()
    del VC     # need this to avoid memory leak, python normally cleans up when going out of scope but the test use of VC and it being C++/ROOT code could mean this does not happen properly (as seen when testing with memory-profiler)
    return

def unpack_fit_coeff(path, nr_of_channels = 24):
    with uproot.open(path) as coef_file:
        #reshapes the param array to have shape (channel, sample, param) = (24,4096,10)
        coef = np.stack(coef_file["coeffs_tree/coeff"].array(library = 'np'))   #stack is needed to convert ndarray of ndarrays to 'normally shaped' array, otherwise you have nested ndarrays
        coef = np.array(np.split(coef, nr_of_channels, axis = 0))
    return coef

def unpack_fit_times(path):
    with uproot.open(path) as fit_file:
        start_time = fit_file["general_tree/startTime"].array(library = "np")[0]
        end_time = fit_file["general_tree/endTime"].array(library = "np")[0]
    return start_time, end_time

def unpack_fit_residuals(path):
    with uproot.open(path) as fit_file:
        vres_dac1, vres_dac2 = fit_file["aveResid_dac1"].member("fX"), fit_file["aveResid_dac2"].member("fX")
        residual_dac1, residual_dac2 = fit_file["aveResid_dac1"].member("fY"), fit_file["aveResid_dac2"].member("fY")
        diff = len(vres_dac1) - len(vres_dac2)
        if diff > 0:
                vres_dac1 = vres_dac1[: -diff]
                residual_dac1 = residual_dac1[: -diff]
        elif diff < 0:
                vres_dac2 = vres_dac2[:diff]
                residual_dac2 = residual_dac2[: diff]
    return np.stack(np.array([vres_dac1, vres_dac2]), axis = -1), np.stack(np.array([residual_dac1, residual_dac2]), axis = -1)

def unpack_vbias(path):
    with uproot.open(path) as bias_file:
        vbias = bias_file["pedestals/vbias[2]"].array(library = "np")
    return vbias

def unpack_adc(path):
    with uproot.open(path) as bias_file:
        adc = bias_file["pedestals/pedestals[24][4096]"].array(library = "np").astype(np.float32)
    return adc

def unpack_time(path):
    with uproot.open(path) as bias_file:
        time = bias_file["pedestals/when"].array(library = "np")
    return time

def unpack_pedestal(run_path):
    pedestal_value = None
    for pedestal_name in ["pedestal.root", "pedestals.root"]:
        try:
            with uproot.open(f"{run_path}/{pedestal_name}") as pedestal:
                pedestal_value = pedestal["pedestals"]["vbias[2]"].array(library = "np")
            break
        except:
            continue
    if pedestal_value is None:
        return None
    return pedestal_value[0]




def fitted_adc(V, param):
    return np.array(np.polyval(param[::-1], V))

def pseudo_fit(V, LAB4Dmax = 2.5, LAB4Dbits = 4095):
    return V *(LAB4Dbits/LAB4Dmax)

def pseudo_fit_inverted(ADC, LAB4Dmax = 2.5, LAB4Dbits = 4095):
    return ADC * (LAB4Dmax/LAB4Dbits)

def fitted_adc_with_res(V, param, res_v, res):
    if len(V) == len(res):
        return np.array(np.polyval(param[::-1], V) + res)
    else:
        res_interpol = interpolate.interp1d(res_v, res)
        return np.array(np.polyval(param[::-1], V) + res_interpol(V))

def rescale_value(vbias, V_ref, DAC):
    vref_index = min(range(len(vbias[:, DAC])), key = lambda i : np.abs(vbias[i, DAC] - V_ref))    

    return vref_index, vbias[vref_index, DAC]

def inverted_function(adc, param, vref, fit_min, fit_max):  #param = [a⁰, ... , a^n]    
    func = np.polynomial.Polynomial(param)
    fit_min_rescaled, fit_max_rescaled = fit_min - vref, fit_max - vref

    inverted_v = (func - adc).roots()
    inverted_v = inverted_v[np.abs(inverted_v.imag) < 1e-5].real

    if adc < 0:
        inverted_v = inverted_v[fit_min_rescaled <= inverted_v]
        inverted_v = inverted_v[inverted_v < 0]

    elif adc >= 0:
        inverted_v = inverted_v[0 <= inverted_v]
        inverted_v = inverted_v[inverted_v <= fit_max_rescaled]

    if inverted_v.size == 0:
        return np.array([fit_max_rescaled])
    
    return(inverted_v)

def cal_adc_to_v(ADC, param, vres, res, starting_window, 
                 accuracy = 100, fit_min_rescaled = -1.3, fit_max_rescaled = 0.7):
    
    # an ADC measurement (typically) is 2048 samples long
    # bias scans always start on window 0
    # the two buffers run from window [0 - 15] and [16 - 31]
    # each window contains 128 samples
    # normal data start on a "random" window


    samples_idx = (128 * starting_window + np.arange(2048)) % 2048
    if starting_window >= 16:
        samples_idx += 2048
    param_reordered = param[samples_idx]
    # for discrete inverse 
    # vsamples = np.arange(fit_min, fit_max, 1/accuracy)

    v_array = []
    for adc, p in zip(ADC, param_reordered):
        # discrete inverse
        # vpowers = np.array([v**np.arange(len(p)) for v in vsamples])
        # adcsamples = np.array([np.sum(vpowers[i] * p) + np.interp(v, vres, res) for i, v in enumerate(vsamples)])
        # adc_borders = adcsamples[np.where((adcsamples < adc))][-1], adcsamples[np.where(adc < adcsamples)][0], 
        
        # full inverse
        # f = polynomial(V) + R(V)
        func = lambda v : np.sum(v**np.arange(len(p)) * p) + np.interp(v, vres, res) - adc
        v = brentq(func, fit_min_rescaled, fit_max_rescaled)
        v_array.appendi(v)
    v_array = np.array(v_array)
    return v_array

def sort_per_time(time_array, array):
    array_sorted = sorted(zip(time_array, array), key = lambda pair: pair[0])
    array_sorted = np.array([el[1] for el in array_sorted])
    return array_sorted
    

@jit(nopython = True)
def numba_fitted_adc(V, param):
    poly_eval = np.zeros(len(V), dtype = np.float32)
    for i, v in enumerate(V):
        var_list = np.array([v**i for i in range(10)], dtype = np.float32)
        evaluated = param.dot(var_list)
        poly_eval[i] = evaluated
    return poly_eval


# function for rescaling the bias scan data to the fit reference frame (which is shifted to set the base pedestal of the station to 0 V)
def rescale_bias_scan(vbias, 
                       V_ref, fit_min, fit_max, 
                       DAC):

    # determining the base pedestal, in the fit scheme this is assumed to be 1.5 but since the steps do not nicely reach exactly 1.5, we will take closest calue to 1.5
    vref_index = min(range(len(vbias[:, DAC])), key = lambda i : np.abs(vbias[i, DAC] - V_ref))
    vref = vbias[vref_index, DAC]

    vbias_rescaled = vbias - vref

    fit_min_rescaled = fit_min - vref
    fit_max_rescaled = fit_max - vref

    return vbias_rescaled, fit_min_rescaled, fit_max_rescaled

def rescale_adc(vbias, adc, V_ref, DAC):
    vref_index = min(range(len(vbias[:, DAC])), key = lambda i : np.abs(vbias[i, DAC] - V_ref))

    adc_rescaled = adc - adc[vref_index]

    return adc_rescaled




# This function is purely for plotting different layers, e.g. mean result of all channels for one time.
# The rescaling is done in a different function

def plot_means(vbias, 
               adc_mean, adc_std,  
               adc_fit_mean, adc_fit_std, 
               res,
               DAC, 
               fit_min, fit_max, 
               title, output, 
               errorscale = 10):

    vbias_fitboundaries = vbias[np.where(fit_min < vbias[:, DAC])]
    vbias_fitboundaries = vbias_fitboundaries[np.where(vbias_fitboundaries[:, DAC] < fit_max)]

    std_bound = adc_std[np.where(fit_min < vbias[:, DAC])]
    std_bound = std_bound[np.where(vbias_fitboundaries[:, DAC] < fit_max)]

    std_fit_bound = adc_fit_std[np.where(fit_min < vbias[:, DAC])]
    std_fit_bound = std_fit_bound[np.where(vbias_fitboundaries[:, DAC] < fit_max)]

    residuals = res[np.where(fit_min < vbias[:, DAC])]
    residuals = residuals[np.where(vbias_fitboundaries[:, DAC] < fit_max)]

    fig, (ax1, ax2, ax3) = plt.subplots(3, gridspec_kw={'height_ratios': [3, 1, 1]}, sharex = True)

    ax1.plot(vbias[:,DAC], adc_fit_mean,  
             color = 'blue', 
             zorder = 0, 
             label = "mean fit")
    
    ax1.fill_between(vbias[:, DAC], 
                     adc_fit_mean - adc_fit_std * 0.5 * errorscale, adc_fit_mean + adc_fit_std * 0.5 * errorscale, 
                     alpha = 0.2, 
                     color = 'blue', 
                     label = f"fit std $\cdot$ {errorscale}")
    
    ax1.errorbar(vbias[::2, DAC], adc_mean[::2],
                yerr=adc_std[::2]*errorscale,
                fmt = "r.", ecolor = 'orange',
                elinewidth= 1.,
                label = f"bias std $\cdot$ {errorscale}")
    
    ax1.set_ylim(np.min(adc_fit_mean - adc_fit_std * 0.5*errorscale) * 1.2, 
                 np.max(adc_fit_mean + adc_fit_std * 0.5*errorscale) * 1.2)
    
    ax1.set_ylabel("ADC")
    ax1.legend(loc = 2)
    x_text, y_text = -0.2, -2000                                                                    # .plt.text takes data coordinates
    ax1.text(x_text, y_text, f'average spread on fits is {np.mean(adc_fit_std):.2f}')               # :.2f rounds to two decimals
    ax1.set_title(f"{title}")

    
    ax2.scatter(vbias_fitboundaries[:, DAC], residuals, color = 'black', marker = '.')
    ax2.plot(vbias_fitboundaries[:, DAC], residuals, color = 'black', lw=1.)
    ax2.set_ylabel("ADC")
    ax2.set_title("Avg residuals")

    ax3.plot(vbias_fitboundaries[:, DAC], std_bound, color = "orange", lw = 1)
    #ax3.plot(vbias_fitboundaries[:, DAC], std_fit_bound, color = "blue", lw = 1)
    ax3.set_xlabel("V$_{bias}$ $(V)$")
    ax3.set_ylabel("ADC")
    ax3.set_title("bias measurement spread")

    fig.savefig(f'{output}', dpi = 500, bbox_inches = "tight")
    return