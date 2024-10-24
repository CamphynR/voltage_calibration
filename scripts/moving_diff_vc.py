import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from utility_functions import read_config, save_to_pickle, unpack_fit_times, unpack_fit_coeff, unpack_fit_residuals, fitted_adc_with_res
from biasParser import biasParser



def calculate_difference_runs(run_path_0, run_path_1, channel = 0, fit_min = -1.29, fit_max = 0.69, v_step = 0.1, DAC = 0):
    print(f"Processing run {os.path.basename(run_path_1)}")
    vc_path_0 = glob.glob(f"{run_path_0}/volCal*")[0]
    vc_path_1 = glob.glob(f"{run_path_1}/volCal*")[0]
    # select start time
    time = unpack_fit_times(vc_path_0)[0]

    vc_0 = unpack_fit_coeff(vc_path_0)
    resv_0, res_0 = unpack_fit_residuals(vc_path_0)
    vc_1 = unpack_fit_coeff(vc_path_1)
    resv_1, res_1 = unpack_fit_residuals(vc_path_1)

    v = np.arange(fit_min, fit_max, v_step)
    adc_0 = np.array([fitted_adc_with_res(v, vc, resv_0[:, DAC], res_0[:, DAC])
             for vc in vc_0[channel, :]])
    adc_1 = np.array([fitted_adc_with_res(v, vc, resv_1[:, DAC], res_1[:, DAC])
             for vc in vc_1[channel, :]])

    vc_diff = np.abs(adc_1 - adc_0)
    vc_diff_mean = np.mean(vc_diff)
    vc_diff_std = np.std(vc_diff)
    return time, vc_diff_mean, vc_diff_std



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = '%(prog)s')
    parser.add_argument("--station", default = 13)
    parser.add_argument("--channel", default = 0)
    parser.add_argument("--config", default = "config.json")
    args = parser.parse_args()

    config = read_config(args.config)   

    run_dir = f"{config['bias_dir']}/station{args.station}"
    biasParser = biasParser(run_dir)
    kwargs = dict(channel = args.channel)
    biasParser.set_function(calculate_difference_runs, kwargs)
    results = np.array(biasParser.double_run())
    
    pickle_dict = dict(time = results[:, 0], vc_diff = results[:, 1], vc_diff_std = results[:, 2], station = args.station, channel = args.channel)
    pickle_path = f"{config['pickle_dir']}/vc_diff/vc_diff_s{args.station}.pickle"
    save_to_pickle(pickle_dict, pickle_path)