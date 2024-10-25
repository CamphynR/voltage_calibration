import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt

from utility_functions import read_pickle, read_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = '%(prog)s')
    parser.add_argument("--pickle")
    parser.add_argument("--config", default = "config.json")
    args = parser.parse_args()
    config = read_config(args.config)

    diff_dict = read_pickle(args.pickle)
    times = diff_dict["time"]
    diff = diff_dict["vc_diff"]
    diff = sorted(zip(times, diff), key = lambda pair : pair[0])
    diff = np.array(diff)[:, 1]
    diff_std = diff_dict["vc_diff_std"]
    diff_std = sorted(zip(times, diff_std), key = lambda pair : pair[0])
    diff_std = np.array(diff_std)[:, 1]
    times = sorted(times)
    times_utc = [datetime.datetime.fromtimestamp(time) for time in times]
    times_utc = [time.strftime("%d  %B %Y") for time in times_utc]

    print(times)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 8), sharey = True, width_ratios = (1, 3))

    ax1.plot(times, diff)
    ax2.plot(times, diff)
    
    cut_idx = np.squeeze(np.where(np.diff(times) > 1e6))

    ax1.set_xlim(times[0], times[cut_idx])
    ax2.set_xlim(times[cut_idx + 1], times[-1])
    ax2.set_ylim(0, 5)

    ax1.set_xticks(times[0:cut_idx:12], times_utc[0:cut_idx:12], rotation =-45, fontsize = "large")
    ax2.set_xticks(times[cut_idx + 1: -1 :12], times_utc[cut_idx + 1:-1:12], rotation =-45, fontsize = "large")

    ax1.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax1.yaxis.tick_left()
    ax2.yaxis.tick_right()

    ax1.set_ylabel(r"$\Delta$ ADC", size = "large")

    # Make the spacing between the two axes a bit smaller
    plt.subplots_adjust(wspace=0.15)

    ax1.grid()
    ax2.grid()

    fig.suptitle("average diff between vc fits (station 13)")
    fig.tight_layout()
    fig_name = f"{config['fig_dir']}/moving_diff_s"
    fig.savefig(fig_name)