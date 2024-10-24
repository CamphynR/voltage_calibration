import argparse
import json
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import uproot

from utility_functions import read_config, read_pickle, find_nr_subplots



if __name__ == '__main__':  
    parser = argparse.ArgumentParser(prog = '%(prog)s')
    parser.add_argument("--pickle")
    parser.add_argument("--config", default = "config.json")
    args = parser.parse_args()

    config = read_config(args.config)

    delta_adc_dict = read_pickle(args.pickle)
    nr_ch = delta_adc_dict["mean"].shape[0]
    nr_subplots = find_nr_subplots(nr_ch)
    fig, axs = plt.subplots(nr_subplots[0], nr_subplots[1], figsize = (32, 24), sharex= True, sharey = True)
    if type(axs) == np.ndarray:
        axs = np.ndarray.flatten(axs)
    else:
        axs = [axs]

    print(delta_adc_dict["mean"].shape)
    for ch, ax in enumerate(axs):
        print(ch)
        ax.plot(delta_adc_dict["vbias"][int(ch > 11)], delta_adc_dict["mean"][ch], label = r"$\Delta$ ADC = bias - linear")
        ax.fill_between(delta_adc_dict["vbias"][int(ch > 11)],
                       delta_adc_dict["mean"][ch] - 0.5 * delta_adc_dict["std"][ch],
                       delta_adc_dict["mean"][ch] + 0.5 * delta_adc_dict["std"][ch],
                       label = r"$\Delta$ ADC = bias - linear", alpha = 0.5)
        ax.hlines(0, -1.5, 0.5, color = "black", ls ="dashed", lw = 2., label = "linear")
        ax.legend(loc = "best")
        ax.set_xlim(-0.3, 0.3)
        ax.set_ylim(-100, 100)
        ax.set_xlabel("voltage / V", size = "x-large")
        ax.set_ylabel("ADC counts", size = "x-large")
        ax.set_title(f"channel {delta_adc_dict['channel'][ch]}")
    fig.tight_layout()
    fig.suptitle(f"Bias scan deviation from linearity (run {delta_adc_dict['run']}, station {delta_adc_dict['station']}, mean over samples")
    if nr_ch == 24:
        ch_name = "all"
    else:
        ch_name = '_'.join(str(c) for c in delta_adc_dict['channel'])
    figname = f"{config['fig_dir']}/vc_dev_from_linearity_run{delta_adc_dict['run']}_s{delta_adc_dict['station']}_ch{ch_name}"
    print(f"Saving as {figname}")
    fig.savefig((figname))