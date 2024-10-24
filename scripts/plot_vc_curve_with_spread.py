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

    vc_dict = read_pickle(args.pickle)
    
    fig, axs = plt.subplots(6, 4, figsize = (32, 24), sharex= True, sharey = True)
    axs = np.ndarray.flatten(axs)

    print(vc_dict["std"])

    for ch, ax in enumerate(axs):
        print(ch)
        ax.plot(vc_dict["v"], vc_dict["mean"][ch], label = r"mean")
        ax.fill_between(vc_dict["v"],
                        vc_dict["mean"][ch] - 2.5 * vc_dict["std"][ch],
                        vc_dict["mean"][ch] + 2.5 * vc_dict["std"][ch],
                        label = r"spread", alpha = 0.5, color = "red")
        ax.legend(loc = "best")
        # ax.set_xlim(-0.2, 0.2)
        # ax.set_ylim(-100, 100)
        ax.set_xlabel("voltage / V", size = "x-large")
        ax.set_ylabel("ADC counts", size = "x-large")
        ax.set_title(f"channel {ch}")
        ax.grid()
    fig.tight_layout()
    fig.suptitle(f"Average vc curve over runs station {vc_dict['station']}")
    
    figname = f"{config['fig_dir']}/vc_curve_s{vc_dict['station']}"
    print(f"Saving as {figname}")
    fig.savefig((figname))