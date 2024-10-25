import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt

from utility_functions import read_config, save_to_pickle, unpack_fit_coeff, unpack_fit_residuals, fitted_adc_with_res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = '%(prog)s')
    parser.add_argument("--station", default = 13)
    parser.add_argument("--config", default = "config.json")
    args = parser.parse_args()

    config = read_config(args.config)

    vc_files = glob.glob(f"{config['bias_dir']}/station{args.station}/run*/vol*")[:50]

    window = config["window"]
    channels = list(range(24))

    v_points = np.arange(window[0], window[1], 0.01)
    adc_mean = np.zeros((24, len(v_points)))
    adc_squares = np.zeros((24, len(v_points)))

    for i, vc_file in enumerate(vc_files):
        print(f"Processing file {i + 1}/{len(vc_files)}")
        vc_coeff = unpack_fit_coeff(vc_file)
        vc_vres, vc_res = unpack_fit_residuals(vc_file)

        adc_vc = np.array([
                        [fitted_adc_with_res(v_points, vc_coeff[ch, s], vc_vres[:,int(ch > 11)], vc_res[:,int(ch > 11)]) for s in range(4096)]
                        for ch in channels])
        adc_mean += np.sum(adc_vc, axis = 1)
        adc_squares += np.sum(adc_vc**2, axis = 1)
    
    adc_mean /= (4096*len(vc_files))
    adc_squares /= (4096*len(vc_files))
    adc_var = adc_squares - adc_mean**2
    adc_std = np.sqrt(adc_var)

    pickle_dict = dict(station = args.station, v = v_points, mean = adc_mean, std = adc_std)
    pickle_name = f"{config['pickle_dir']}/vc_curves/vc_curv_spread_s{args.station}"
    save_to_pickle(pickle_dict, pickle_name)