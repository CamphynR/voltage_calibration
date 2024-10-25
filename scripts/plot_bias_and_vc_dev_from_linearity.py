import argparse
import json
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import uproot

from utility_functions import unpack_vbias, unpack_adc 

def read_pickle(pickle_path):
    with open(pickle_path, "rb") as pickle_file:
        array = pickle.load(pickle_file)
    return array



if __name__ == '__main__':  
    parser = argparse.ArgumentParser(prog = '%(prog)s')
    parser.add_argument("--bias_pickle")
    parser.add_argument("--vc_pickle")
    parser.add_argument("--config", default = "config.json")
    args = parser.parse_args()

    with open(args.config, "r") as config_json:
        config = json.load(config_json)

    delta_adc_bias_dict = read_pickle(args.bias_pickle)  
    delta_adc_vc_dict = read_pickle(args.vc_pickle)  

    idx = np.abs(delta_adc_bias_dict["vbias"][:, 0] - 1.5).argmin()
    offset = delta_adc_bias_dict["mean"][0, idx]
    
    fig, ax = plt.subplots(1, 1, figsize = (12, 8))
    ax.errorbar(delta_adc_bias_dict["vbias"][:, 0], delta_adc_bias_dict["mean"], yerr = delta_adc_bias_dict["std"], label = r"$\Delta$ ADC = bias - linear", color = "blue")
    ax.errorbar(delta_adc_vc_dict["vbias"][:, 0] + 1.5, delta_adc_vc_dict["mean"] + offset, yerr = delta_adc_vc_dict["std"], label = r"$\Delta$ ADC = calibrated - linear", color = "red")
    ax.hlines(0, 2.5, 0, color = "black", ls ="dashed", lw = 2., label = "linear")
    ax.legend(loc = "best")
    ax.set_xlim(0.2, 2.2)
    ax.set_ylim(-100, 100)
    ax.set_xlabel("voltage / V", size = "x-large")
    ax.set_ylabel("ADC counts", size = "x-large")
    fig.suptitle(f"Deviation from linearity for both raw bias scan and calibration (run {delta_adc_bias_dict['run']}, station {delta_adc_bias_dict['station']}, mean over channel {delta_adc_bias_dict['channel']})")

    fig.savefig(f"{config['fig_dir']}/bias_cal_dev_from_linearity_run{delta_adc_bias_dict['run']}_s{delta_adc_bias_dict['station']}_ch{''.join(str(c) for c in delta_adc_bias_dict['channel'])}")