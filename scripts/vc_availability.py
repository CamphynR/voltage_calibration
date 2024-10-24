import argparse
import glob
import datetime
import numpy as np
import matplotlib.pyplot as plt

from utility_functions import read_config

def get_time_from_filename(filename):
    """
    assumes filename to be of the form volCalConsts_pol9_s{station}_{start_time}-{end_time}.root
    """
    time = filename.split("_")[-1]
    time = time.split("-")[0]
    return int(time)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = '%(prog)s')
    parser.add_argument("--config", default = "config.json")
    args = parser.parse_args()
    config = read_config(args.config)

    stations = [11, 12, 13, 21, 22, 23, 24]

    
    times = []
    for station in stations:
        vc_files = glob.glob(f"{config['bias_dir']}/station{station}/run*/vol*")
        vc_files += glob.glob(f"{config['bias_dir']}/season22_scan_testing/station{station}/vol*s{station}*")
        vc_files = sorted(vc_files, key = lambda filename : filename.split("/")[-1])
        times_station = [get_time_from_filename(vc_file) for vc_file in vc_files]
        times.append(times_station)

    times_utc = [[datetime.datetime.fromtimestamp(time) for time in times_station] for times_station in times]
    times_utc = [[time.strftime("%d-%b-%Y") for time in times_station] for times_station in times_utc]

    fig, axs = plt.subplots(len(stations), 1, figsize = (14, 12), sharex = True)
    nr_of_ticks = 12
    axs = np.ndarray.flatten(axs)
    for i, ax in enumerate(axs):
        ax.scatter(times[i], np.ones_like(times[i]), label = f"vc available: {len(times[i])} scans", color = "blue", marker = "s")
        ax.set_title(f"Station {stations[i]}")
        
        time_2023 = 1672534800
        time_utc_2023 = datetime.datetime.fromtimestamp(time_2023).strftime("%d-%b-%Y")
        ax.vlines(time_2023, 0, 1.5, ls = "dashed", color = "red", label = "1 Jan 2023")
        
        ax.set_ylim(0.9, 1.1)
        ax.get_yaxis().set_visible(False)
        ax.legend(loc = 2)
    min_time = min(min(times))
    min_time_utc = datetime.datetime.fromtimestamp(min_time).strftime("%d-%b-%Y")
    step = int(len(times[-1])/nr_of_ticks)
    axs[-1].set_xticks([min_time] + [time_2023] + times[-1][::step], [min_time_utc] + [time_utc_2023] + times_utc[-1][::step], rotation = -30, ha = "left")
    # for ax in axs:
    #     ax.set_xlim(time_2023 + 2.5 * 2629743, None)

    fig.suptitle("Availablity of voltage calibration per station for 2022 - 2023", size = "x-large", weight = "bold")
    fig.tight_layout()
    fig.savefig("/user/rcamphyn/voltage_calibration/test.png")