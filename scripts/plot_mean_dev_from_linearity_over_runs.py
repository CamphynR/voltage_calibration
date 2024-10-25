import argparse
import datetime
import json
import numpy as np
import matplotlib.pyplot as plt

from utility_functions import read_pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = '%(prog)s')
    parser.add_argument("-p", "--pickle_path")
    parser.add_argument("--config", default = "config.json")
    args = parser.parse_args()

    with open(args.config, "r") as config_json:
        config = json.load(config_json)

    adc_dict = read_pickle(args.pickle_path)
    vc_times = np.array(adc_dict["vc_times"])[:, 0]
    # only use start time of bias scan on x-axis
    adc = sorted(zip(vc_times, adc_dict["delta_adc"][:, 0]), key = lambda pair : pair[0])
    adc = np.array(adc)[:, 1]
    adc_std = sorted(zip(vc_times, adc_dict["delta_adc"][:, 1]), key = lambda pair : pair[0])
    adc_std = np.array(adc_std)[:, 1]
    vc_times = sorted(vc_times)
    vc_times_start_utc = [datetime.datetime.fromtimestamp(vc_time) for vc_time in vc_times]
    vc_times_start_utc = [vc_time_utc.strftime("%d %B %Y") for vc_time_utc in vc_times_start_utc]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 8), sharey = True, width_ratios = [1, 3])
    ax1.plot(vc_times,
             adc,
             color = "blue")
    ax1.fill_between(vc_times,
                     adc - 0.5* adc_std,
                     adc + 0.5* adc_std,
                     color = "blue",
                     alpha = 0.5)
    
    ax2.plot(vc_times,
             adc,
             color = "blue")
    ax2.fill_between(vc_times,
                     adc - 0.5* adc_std,
                     adc + 0.5* adc_std,
                     color = "blue",
                     alpha = 0.5)
    
    cut_idx = np.squeeze(np.where(np.diff(vc_times) > 1e6))

    ax1.set_xlim(vc_times[0], vc_times[cut_idx])
    ax2.set_xlim(vc_times[cut_idx + 1], vc_times[-1])

    ax1.set_xticks(vc_times[0:cut_idx:12], vc_times_start_utc[0:cut_idx:12], rotation =-45, fontsize = "large")
    ax2.set_xticks(vc_times[cut_idx + 1: -1 :12], vc_times_start_utc[cut_idx + 1:-1:12], rotation =-45, fontsize = "large")
    


    ax1.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax1.yaxis.tick_left()
    ax2.yaxis.tick_right()

    # Make the spacing between the two axes a bit smaller
    plt.subplots_adjust(wspace=0.15)

    ax1.grid()
    ax2.grid()

    # ax1.set_xticks(vc_times[0:cut_idx:12], vc_times_start_utc[0:cut_idx:12], rotation =-45, fontsize = "large")
    # ax2.set_xticks(vc_times[cut_idx:-1:12], vc_times_start_utc[cut_idx:-1:12], rotation =-45, fontsize = "large")
    fig.text(0.5, -0.1, "Time of bias scan / UTC", ha = "center", size = "large")
    ax1.set_ylabel(r"$\Delta ADC$", size = "large")
    
    fig.suptitle(f"Mean absolute deviation from linearity over time station {adc_dict['station']} channel {''.join(str(c) for c in adc_dict['channel'])}")

    
    figname = f"{config['fig_dir']}/mean_dev_from_linearity_over_runs_s{adc_dict['station']}_ch{''.join(str(c) for c in adc_dict['channel'])}"
    fig.savefig(figname, bbox_inches = "tight")