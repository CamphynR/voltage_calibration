import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

from utility_functions import read_pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = '%(prog)s')
    parser.add_argument("--pickle")
    parser.add_argument("--config", default = "config.json")
    args = parser.parse_args()

    with open(args.config, "r") as config_json:
        config = json.load(config_json)

    DAC = 0

    ped_dict = read_pickle(args.pickle)
    station = ped_dict["station"]
    times = ped_dict["pedestal_times"]
    pedestals = np.squeeze(ped_dict["pedestals"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 8))
    ax1.plot(times, pedestals)

    fig_path = f"{config['fig_dir']}/pedestals_over_time_s{station}"
    fig.savefig(fig_path)