import argparse
import csv
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from utility_functions import make_folder, read_pickle, read_config

def find_2023_time_idx(times):
    epoch_2023 = 1672534800
    for i, t in enumerate(times):
        if t > epoch_2023:
            return i
        

def read_temp_from_csv(csv_file, newline = ''):
    with open(csv_file, "r") as f:
        reader = csv.reader(f, delimiter=",")
        reader.__next__()
        temp_times = []
        temp = []
        for row in reader:
            temp_times.append(float(row[0]))
            temp.append(float(row[1]))
        return temp_times, temp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = '%(prog)s')
    parser.add_argument("pickle")
    parser.add_argument("--exclude_2022", action = "store_true")
    parser.add_argument("--config", default = "config.json")
    args = parser.parse_args()
    config = read_config(args.config)
    temp_file = "/home/ruben/Documents/projects/RNO-G_calibration_V2/temperatures/housekeepingdata_st23_2022-09-20to2023-10-31_v2_single.csv"
    temp_times, temp = read_temp_from_csv(temp_file)

    vc_drift_dict = read_pickle(args.pickle)
    station = vc_drift_dict["station"]
    times = np.array(vc_drift_dict["times"])
    window = vc_drift_dict['v'][0], vc_drift_dict['v'][-1]
    adc_diff = np.array(vc_drift_dict["adc_diffs"])
    adc_diff_std = np.std(adc_diff, axis = -1)
    adc_diff = np.mean(adc_diff, axis = -1)

    times_utc = [datetime.datetime.fromtimestamp(t) for t in times]
    times_utc = [t.strftime("%d/%m/%y") for t in times_utc]

    nr_of_ticks = 10

    # convert linearly to mV for clarity in plot
    adc_diff = adc_diff * (2500/4095)
    adc_diff_std = adc_diff_std * (2500/4095)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (18, 8), sharey = True, width_ratios=[1, 6])
    
    for c in range(24):   
        i = 0
        for ax in [ax1, ax2]:
            if i == 0:
                legend_handle = ax.plot(times, adc_diff[:, c],
                lw = 2.,
                color = "blue",
                alpha = 0.6,
                label = "channels")
            else:
                ax.plot(times, adc_diff[:, c],
                            lw = 2.,
                            color = "blue",
                            alpha = 0.6,)
                # ax.fill_between(times, adc_diff[:, c],
                #                 adc_diff[:, c] - 0.5 * adc_diff_std[:, c],
                #                 adc_diff[:, c] + 0.5 * adc_diff_std[:, c],
                #                 alpha = 0.6,
                #                 color = "blue"
                #                 )         
    
            # line to indicate 1 jan 2023
            # idx = find_2023_time_idx(times)
            # ax.vlines(idx, 0, np.max(adc_diff[:, :]),
            #                 ls = "dashed", color = "red", label = "2022")
            
            
            tick_step = int(len(times_utc)/10)
            ax.set_xticks(times[::tick_step], times_utc[::tick_step], rotation = -45, size = "xx-large")
            # ax.set_xticks(range(len(times_utc))[::tick_step], times_utc[::tick_step], rotation = -45, size = "xx-large")
            ax.tick_params(axis = "y", labelsize = "xx-large")
            ax.grid()

            # if args.exclude_2022:
            #     ax.set_xlim(idx, None)
            #     ax.set_ylim(None, 1.1 * np.max(adc_diff[idx + 1:,:]))

            

            axsec = ax.twinx()
            axsec.plot(temp_times[::100], temp[::100], label = "temperature", color = 'red', lw = 1.)
   
            if i == 0:
                axsec.spines["right"].set_visible(False)
                axsec.yaxis.set_major_locator(ticker.NullLocator())
                i+=1
            elif i == 1:
                legend = ax.legend(handles = legend_handle, loc = "best", fontsize = "xx-large", facecolor = "white", framealpha = 1.)
                legend.remove()
                axsec.add_artist(legend)
                axsec.legend(loc = (0.8155, 0.83), fontsize = "xx-large", framealpha = 1.)
                axsec.spines["left"].set_visible(False)        
                axsec.set_ylabel("Temperature / C", size = "xx-large")
                axsec.tick_params(axis = "y", labelsize = "xx-large")

    ax1.set_xlim(times[0] - 1000000,times[0] + 1000000)
    ax2.set_xlim(times[0] + 17500000, times[-1] + 100000)
    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    fig.text(0.5, -0.02, "Date / UTC", size = "xx-large", ha = "center")
    ax1.set_ylabel("Diff with previous calibration / mV", size = "xx-large")
       


    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # bottom left
    ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)        # top left


    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (-d, +d), **kwargs)  # bottom-right
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # top right

    fig.suptitle(f"Voltage calibration drift, station {station}", size = "xx-large")
    fig.tight_layout()
    
    if "no_ref" in args.pickle:
        fig_dir = f"{config['fig_dir']}/vc_drift_no_ref/"
    else:
        fig_dir = f"{config['fig_dir']}/vc_drift_ref/"
    make_folder(fig_dir)
    figname = args.pickle.split("/")[-1]
    if args.exclude_2022:
        figname += "_no_2022"
    figname += "_single"
    figname += "_stacked"
    figname+=".png"
    print(fig_dir + figname)
    fig.savefig(fig_dir + figname, bbox_inches = "tight")