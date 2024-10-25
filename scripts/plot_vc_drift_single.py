import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt

from utility_functions import make_folder, read_pickle, read_config

def find_2023_time_idx(times):
    epoch_2023 = 1672534800
    for i, t in enumerate(times):
        if t > epoch_2023:
            return i

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = '%(prog)s')
    parser.add_argument("pickle")
    parser.add_argument("--exclude_2022", action = "store_true")
    parser.add_argument("--config", default = "config.json")
    args = parser.parse_args()
    config = read_config(args.config)

    

    vc_drift_dict = read_pickle(args.pickle)
    station = vc_drift_dict["station"]
    times = np.array(vc_drift_dict["times"])
    window = vc_drift_dict['v'][0], vc_drift_dict['v'][-1]
    adc_diff = np.array(vc_drift_dict["adc_diffs"])
    adc_diff_std = np.std(adc_diff, axis = -1)
    adc_diff = np.mean(adc_diff, axis = -1)
    print(times.shape)
    print(adc_diff.shape)
    print(vc_drift_dict['v'])

    times_utc = [datetime.datetime.fromtimestamp(t) for t in times]
    times_utc = [t.strftime("%d/%m/%y") for t in times_utc]
    print(times_utc[0])

    nr_of_ticks = 10

    fig, ax = plt.subplots(figsize = (32, 24))
    for c in range(24):
        ax.plot(adc_diff[:, c],
                    color = "blue",
                    label = f"channel {c}")
        ax.fill_between(range(len(adc_diff[:, c])), adc_diff[:, c],
                            adc_diff[:, c] - 0.5 * adc_diff_std[:, c],
                            adc_diff[:, c] + 0.5 * adc_diff_std[:, c],
                            color = "blue")
    # line to indicate 1 jan 2023
    idx = find_2023_time_idx(times)
    ax.vlines(idx, 0, np.max(adc_diff[:, :]),
                    ls = "dashed", color = "red", label = "2022")
    
    ax.legend(loc = "upper left", fontsize = "xx-large", bbox_anchor = (1.05, 1.), ncols = 2)
    tick_step = int(len(times_utc)/10)
    ax.set_xticks(range(len(times_utc))[::tick_step], times_utc[::tick_step], rotation = -45, size = "xx-large")
    ax.tick_params(axis = "y", labelsize = "xx-large")
    ax.grid()
    ax.set_title(f"channel {c}", size = 24)

    if args.exclude_2022:
        ax.set_xlim(idx, None)
        ax.set_ylim(None, 1.1 * np.max(adc_diff[idx + 1:,:]))

    ax2 = ax.secondary_yaxis("right",
                                functions = (lambda ADC : ADC * (2.5/4095) * 1000, lambda V : V/1000 * (4095/2.5)))
    ax2.tick_params(axis = "y", labelsize = "xx-large")
    ax2.set_ylabel("mV", rotation = -90, size = 18)
    # Save just the portion _inside_ the second axis's boundaries
    # extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

    # figname = f"{config['fig_dir']}/vc_drift_ref/vc_drift_s{station}_ch{c}"
    # Pad the saved area by 10% in the x-direction and 20% in the y-direction
    # fig.savefig(figname, bbox_inches=extent.expanded(1.1, 1.2))

    fig.text(0.5, -0.02,
             "Date / UTC", size = 32, ha = "center")
    fig.text(-0.02, 0.5,
             "Diff with previous calibration / ADC", size = 32, va = "center", rotation = "vertical")
    fig.text(1.02, 0.5,
             "Diff with previous calibration / mV", size = 32, va = "center", rotation = -90)
    fig.text(0.5, 1.02,
             f"Voltage calibration drift, station {station}, window [{window[0]:.2f}, {window[1]:.2f}] V", ha = "center", size = 32)
    fig.tight_layout()
    
    if "no_ref" in args.pickle:
        fig_dir = f"{config['fig_dir']}/vc_drift_no_ref/"
    else:
        fig_dir = f"{config['fig_dir']}/vc_drift_ref/"
    make_folder(fig_dir)
    figname = args.pickle.split("/")[-1]
    if args.exclude_2022:
        figname += "_no_2022"
    figname += "single"
    figname+=".png"
    print(fig_dir + figname)
    fig.savefig(fig_dir + figname, bbox_inches = "tight")