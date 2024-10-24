import argparse
import os
import json
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import uproot

from utility_functions import read_config, unpack_vbias, unpack_adc
from voltageCalibration import voltageCalibration


def save_to_pickle(array, pickle_path):
    with open(pickle_path, "wb") as pickle_file:
        pickle.dump(array, pickle_file)
    return



if __name__ == '__main__':  
    parser = argparse.ArgumentParser(prog = '%(prog)s')
    parser.add_argument("--station", default = 13)
    parser.add_argument("--channel", default = ["all"], nargs="+")
    parser.add_argument("--no_window", action = "store_true")
    parser.add_argument("--config", default = "config.json")
    args = parser.parse_args()

    config = read_config(args.config)

    if args.channel == ["all"]:
        args.channel = list(range(24))
    else:
        args.channel = [int(c) for c in args.channel]
    
    # linear factors
    vref = 1.5
    adc_to_v = (2.5)/4095
    v_to_adc = 4095/2.5

    if args.no_window:
        vmin = -1.3
        vmax = 0.7
    else:
        vmin = config['window'][0]
        vmax = config['window'][1]
    
    run_paths = glob.glob(f"{config['bias_dir']}/station{args.station}/run*")
    run_paths = sorted(run_paths)
    vc_times = []
    delta_adc = []
    for run_path in run_paths:
        run_name = os.path.basename(run_path)
        print(f'Processing {run_name}')

        bias_path = f"{run_path}/bias_scan.root"
        vc_path = glob.glob(f"{run_path}/volCalConst*.root")[0]
        vc = voltageCalibration(vc_path)
        vc_time = vc.get_times()
        vc_times.append(vc_time)
        
        vbias = unpack_vbias(bias_path)
        # rescale to pedestal
        vbias = vbias - 1.5
        # only keep values within boundaries
        vbias = np.squeeze(
            [vbias[(np.where((vmin < vbias[:, 0]) & (vbias[:, 0] < vmax))), 0],
             vbias[(np.where((vmin < vbias[:, 1]) & (vbias[:, 1] < vmax))), 1]]).T
        adc = unpack_adc(bias_path)

        # adc in shape (channels, samples, ped)
        adc_vc = np.array([
                            [vc.get_fit_curve(vbias[:, int(ch > 11)], ch, s) for s in range(2048)]
                            for ch in args.channel])
        adc_linear = vbias * v_to_adc

        delta_adc_run = np.array([[adc_vc[ch, s, :] - adc_linear[:, int(ch > 11)] for s in range(2048)] for ch in args.channel])
        delta_adc_mean = np.mean(np.abs(delta_adc_run), axis = -1)
        delta_adc_std = np.std(np.abs(delta_adc_run), axis = -1)
        delta_adc.append([delta_adc_mean, delta_adc_std])
    delta_adc = np.array(delta_adc)

    ch_name = f"_ch{''.join(str(c) for c in args.channel)}"
    if len(args.channel) == 24:
        ch_name = "_chall"
    pickle_name = f"{config['pickle_dir']}/delta_adc/delta_adc_vc_s{args.station}" + ch_name
    if args.no_window:
        pickle_name += "_no_window"
    else:
        pickle_name += f"_w_{config['window'][0]}_{config['window'][1]}"
    pickle_dict = dict(station = args.station, channel = args.channel, vc_times = vc_times,
                    vbias = vbias, delta_adc = delta_adc, window = (config['window'][0], config['window'][1]))
    save_to_pickle(pickle_dict,
                   pickle_name) 