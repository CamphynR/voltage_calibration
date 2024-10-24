import argparse
import datetime
import json
import numpy as np
import matplotlib.pyplot as plt

from utility_functions import read_pickle

def combine_std(stds):
    std_total = 1 / (np.sum([1/std for std in stds]))
    return std_total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = '%(prog)s')
    parser.add_argument("pickle")
    parser.add_argument("--config", default = "config.json")
    args = parser.parse_args()

    with open(args.config, "r") as config_json:
        config = json.load(config_json)

    adc_dict = read_pickle(args.pickle)
    channels = adc_dict['channel']
    vc_times = np.array(adc_dict["vc_times"])[:, 0]
    # only use start time of bias scan on x-axis
    adc = adc_dict['delta_adc'][:, 0]
    adc = np.mean(adc, axis = -1)
    print(adc.shape)
    adc_std = adc_dict["delta_adc"][:, 1]
    adc_std = np.array([[combine_std(std[ch, :]) for ch in channels] for std in adc_std])
    print(adc_std.shape)

    vc_times = sorted(vc_times)
    vc_times_start_utc = [datetime.datetime.fromtimestamp(vc_time) for vc_time in vc_times]
    vc_times_start_utc = [vc_time_utc.strftime("%d %B %Y") for vc_time_utc in vc_times_start_utc]

    nr_ticks = 10
    tick_steps = int(len(adc)/nr_ticks) 

    fig, axs = plt.subplots(6, 4, figsize = (32, 24), sharex = True, sharey = True)
    axs = np.ndarray.flatten(axs)
    for ch in channels:
        axs[ch].plot(adc[:, ch],
                color = "blue")
        axs[ch].fill_between(range(len(adc)), adc[:, ch],
                            adc[:, ch] - 0.5* adc_std[:, ch],
                            adc[:, ch] + 0.5* adc_std[:, ch],
                            color = "blue",
                            alpha = 0.5)

        axs[ch].set_xticks(list(range(len(adc)))[::tick_steps], vc_times_start_utc[::tick_steps], rotation =-45, fontsize = "large")
        axs[ch].tick_params(axis = "y", labelsize = 18)
        axs[ch].grid()
        axs[ch].set_title(f"Channel {ch}", size = 18)

        if (ch - 3.) %    4. == 0:
            ax2 = axs[ch].secondary_yaxis("right",
                                        functions = (lambda ADC : ADC * (2.5/4095) * 1000, lambda V : V/1000 * (4095/2.5)))
            ax2.tick_params(axis = "y", labelsize = "xx-large")
            ax2.set_ylabel("mV", rotation = -90, size = 18)

    
    fig.text(0.5, -0.02, "Time of bias scan / UTC", ha = "center", size = 32)
    fig.text(-0.02, 0.5, "Deviation from linear / ADC", ha = "center", size = 32, rotation = "vertical")
    fig.suptitle(f"Mean absolute deviation from linearity over time station {adc_dict['station']}, window {adc_dict['window']}", size = 32)
    fig.tight_layout()

    if len(channels) == 24:
        ch_name = "_chall"
    else:
        ch_name = f"_ch{''.join(str(c) for c in adc_dict['channel'])}"
    figname = f"{config['fig_dir']}/mean_dev_from_linearity_over_runs_s{adc_dict['station']}" + ch_name
    fig.savefig(figname, bbox_inches = "tight")