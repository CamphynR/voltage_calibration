import argparse
import json
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import uproot

from utility_functions import save_to_pickle, unpack_vbias, unpack_adc, unpack_pedestal



if __name__ == '__main__':  
    parser = argparse.ArgumentParser(prog = '%(prog)s')
    parser.add_argument("--run", default = 1092)
    parser.add_argument("--station", default = 13)
    parser.add_argument("--channel", default = [0], nargs="+")
    parser.add_argument("--config", default = "config.json")
    args = parser.parse_args()

    with open(args.config, "r") as config_json:
        config = json.load(config_json)

    
    if args.channel == ["all"]:
        args.channel = list(range(24))
        ch_name = "all"
    else:
        args.channel = [int(ch) for ch in args.channel]
        ch_name = '_'.join(str(c) for c in args.channel)
    
    # linear factors
    adc_to_v = (2.5)/4095
    v_to_adc = 4095/2.5

    
    run_path = glob.glob(f"{config['bias_dir']}/station{args.station}/run{args.run}")[0]
    bias_path = f"{run_path}/bias_scan.root"
    vbias = unpack_vbias(bias_path)
    adc = unpack_adc(bias_path)

    v_ped = unpack_pedestal(run_path)
    v_ped_idx = [np.abs([vbias[:, DAC] - v_ped[DAC]]).argmin() for DAC in range(2)]
    adc_ped = np.array([adc[v_ped_idx[int(ch > 11)], ch, :] for ch in range(24)])

    vbias = vbias - v_ped
    adc = adc - adc_ped

    adc_linear = vbias * v_to_adc
    delta_adc = np.array([[adc[:, ch, s] - adc_linear[:, int(ch > 11)] for s in range(2048)] for ch in args.channel])
    delta_adc_mean = np.mean(delta_adc, axis = 1)
    delta_adc_std = np.std(delta_adc, axis = 1)

   
    pickle_name = f"{config['pickle_dir']}/delta_adc/delta_adc_run{args.run}_s{args.station}_ch{ch_name}"
    pickle_dict = dict(run = args.run, station = args.station, channel = args.channel,
                       vbias = vbias, mean = delta_adc_mean, std = delta_adc_std)
    save_to_pickle(pickle_dict,
                   pickle_name)