import argparse
import json
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import uproot

from utility_functions import read_config, unpack_vbias, unpack_adc, unpack_fit_coeff, unpack_fit_residuals, fitted_adc_with_res, unpack_pedestal



def save_to_pickle(array, pickle_path):
    with open(pickle_path, "wb") as pickle_file:
        pickle.dump(array, pickle_file)
    return



if __name__ == '__main__':  
    parser = argparse.ArgumentParser(prog = '%(prog)s')
    parser.add_argument("--run", default = 1092)
    parser.add_argument("--station", default = 13)
    parser.add_argument("--channel", default = [0], nargs="+")
    parser.add_argument("--config", default = "config.json")
    args = parser.parse_args()

    config = read_config(args.config)
    
    if args.channel == ["all"]:
        args.channel = list(range(24))
        ch_name = "all"
    else:
        args.channel = [int(ch) for ch in args.channel]
        ch_name = '_'.join(str(c) for c in args.channel)
    
    # linear factor
    v_to_adc = 4095/2.5

    # limits used in VC parameter fit
    v_min = 0.2
    v_max = 2.2



    
    run_path = glob.glob(f"{config['bias_dir']}/station{args.station}/run{args.run}")[0]
    bias_path = f"{run_path}/bias_scan.root"
    vc_path = glob.glob(f"{run_path}/volCalConst*.root")[0]

    vbias = unpack_vbias(bias_path)
    adc = unpack_adc(bias_path)
    vc_coeff = unpack_fit_coeff(vc_path)
    vc_vres, vc_res = unpack_fit_residuals(vc_path)
    ped = unpack_pedestal(run_path)

    # per DAC
    vbias[:, 0] = vbias[:, 0] - ped[0]
    vbias[:, 1] = vbias[:, 1] - ped[1]
    # only look within fit boundaries
    vbias = np.squeeze(
            [vbias[(np.where((-1.3 < vbias[:, 0]) & (vbias[:, 0] < 0.7))), 0],
             vbias[(np.where((-1.3 < vbias[:, 1]) & (vbias[:, 1] < 0.7))), 1]])

    adc_vc = np.array([
                       [fitted_adc_with_res(vbias[int(ch > 11)], vc_coeff[ch, s], vc_vres[:,int(ch > 11)], vc_res[:,int(ch > 11)]) for s in range(4096)]
                       for ch in args.channel])

    adc_linear = vbias * v_to_adc
    delta_adc = np.array([[adc_vc[ch, s, :] - adc_linear[int(ch > 11)] for s in range(2048)] for ch in args.channel])
    delta_adc_mean = np.mean(delta_adc, axis = (1))
    delta_adc_std = np.std(delta_adc, axis = (1))

    pickle_name = f"{config['pickle_dir']}/delta_adc/delta_adc_vc_run{args.run}_s{args.station}_ch{ch_name}"
    pickle_dict = dict(run = args.run, station = args.station, channel = args.channel,
                       vbias = vbias, mean = delta_adc_mean, std = delta_adc_std)
    save_to_pickle(pickle_dict,
                   pickle_name)