import argparse
import csv
import datetime
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages

from utility_functions import make_folder, read_pickle, read_config


def find_2023_time_idx(times):
    epoch_2023 = 1672534800
    for i, t in enumerate(times):
        if t > epoch_2023:
            return i


def read_temp_from_csv(csv_file, newline=''):
    with open(csv_file, "r") as f:
        reader = csv.reader(f, delimiter=",")
        reader.__next__()
        temp_times = []
        temp = []
        for row in reader:
            temp_times.append(float(row[0]))
            temp.append(float(row[1]))
        return temp_times, temp


def find_skip(times):
    """
    Function to find index on which to change subplot
    """
    limit = datetime.timedelta(days=6*30)
    limit_utc = limit.total_seconds()
    max_i = 0
    max_diff = 0
    for i, diff in enumerate(np.diff(times)):
        if diff > limit_utc:
            if diff > max_diff:
                max_diff = diff
                max_i = i
    return max_i

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='%(prog)s')
    parser.add_argument("pickle")
    parser.add_argument("--exclude_2022", action="store_true")
    parser.add_argument("--show_outliers", action="store_true")
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    config = read_config(args.config)

    vc_drift_dict = read_pickle(args.pickle)
    station = vc_drift_dict["station"]
    times = np.array(vc_drift_dict["times"])
    window = vc_drift_dict['v'][0], vc_drift_dict['v'][-1]
    adc_diff = np.array(vc_drift_dict["adc_diffs"])
    adc_diff_std = np.std(adc_diff, axis=-1)
    adc_diff = np.mean(adc_diff, axis=-1)

    times_utc = [datetime.datetime.fromtimestamp(t) for t in times]
    times_utc = [t.strftime("%d/%m/%y") for t in times_utc]

    temp_dir = "/user/rcamphyn/voltage_calibration/temperatures"
    temp_file = glob.glob(f"{temp_dir}/housekeepingdata*st{station}*")[0]
    temp_times, temp = read_temp_from_csv(temp_file)

    channel_mapping = [[0, 1, 2, 3], [9, 10, 22, 23], [5, 6, 7], [4, 8, 11, 21], [12, 14, 15, 17, 18, 20], [13, 16, 19]]
    channel_classes = ["phased array", "low Vpols", "upper Vpols", "Hpols", "downward LPDA", "upward LPDA"]

    # cut off anything bigger than this limit (mV)
    outlier_limit = 30

    nr_of_ticks = 10

    # convert linearly to mV for clarity in plot
    adc_diff = adc_diff * (2500/4095)
    adc_diff_std = adc_diff_std * (2500/4095)

    if "no_ref" in args.pickle:
        fig_dir = f"{config['fig_dir']}/vc_drift_no_ref/"
    else:
        fig_dir = f"{config['fig_dir']}/vc_drift_ref/"
    make_folder(fig_dir)
    figname = args.pickle.split("/")[-1]
    if args.show_outliers:
        figname += "_with_outliers"
    if args.exclude_2022:
        figname += "_no_2022"
    figname += "_single"
    figname += ".pdf"
    figpdf = PdfPages(fig_dir + figname)
    print(f"Saving as {fig_dir + figname}")

    for class_idx, channel_class in enumerate(channel_classes):
        channel_idxs = channel_mapping[class_idx]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), sharey=True, width_ratios=[1, 6])

        i = 0
        adc_max = 0
        for ax in [ax1, ax2]:
            for c in channel_idxs:
                if args.show_outliers:
                    times_ch = times
                    adc_diff_ch = adc_diff[:, c]
                    adc_diff_std_ch = adc_diff_std[:, c]
                else:
                    snipped_idx = np.nonzero(adc_diff[:, c] < outlier_limit)
                    outlier_idx = np.nonzero(adc_diff[:, c] > outlier_limit)
                    times_ch = times[snipped_idx]
                    adc_diff_ch = np.squeeze(adc_diff[snipped_idx, c])
                    adc_diff_std_ch = np.squeeze(adc_diff_std[snipped_idx, c])

                a = ax.scatter(times_ch, adc_diff_ch,
                               label=f"channel {c}",
                               lw=2.)
                col = a.get_edgecolor()
                if i == 1:
                    ax.fill_between(times_ch,
                                    adc_diff_ch - adc_diff_std_ch,
                                    adc_diff_ch + adc_diff_std_ch,
                                    alpha=0.5,
                                    color=col)
                adc_max = np.max([adc_max, np.max(adc_diff_ch)])
#            ax.set_yscale("log")
            # add points to indicate where outliers lay outside of plot
            if not args.show_outliers:
                outlier_x = times[outlier_idx]
                outlier_y = 1.2*adc_max
                outlier_y = np.repeat(outlier_y, len(outlier_x))
                ax.scatter(outlier_x, outlier_y, marker="s", linewidths=3., color="black")
            # line to indicate 1 jan 2023
            idx = find_2023_time_idx(times)
            # ax.vlines(idx, 0, np.max(adc_diff[:, :]),
            #                 ls = "dashed", color = "red", label = "2022")

            tick_step = int(len(times_utc)/10)
            ax.set_xticks(times[::tick_step], times_utc[::tick_step], rotation=-45, size="xx-large")
            ax.tick_params(axis="y", labelsize="xx-large")
            ax.grid()

            if args.exclude_2022:
                ax.set_xlim(idx, None)
                ax.set_ylim(None, 1.1 * np.max(adc_diff[idx + 1:, :]))

            axsec = ax.twinx()
            axsec.plot(temp_times[::100], temp[::100], label="temperature", color='red', lw=3.)
            if i == 0:
                axsec.spines["right"].set_visible(False)
                axsec.yaxis.set_major_locator(ticker.NullLocator())
                i += 1
            elif i == 1:
                if not args.show_outliers:
                    #hacky way to get single label on legend
                    ax.scatter(outlier_x, outlier_y, marker="s", linewidths=3., color="black",
                               label=f"Outlier > {outlier_limit} mV")
                legend = ax.legend(loc="upper left", fontsize="xx-large", facecolor="white", framealpha=1.)
                legend.remove()
                axsec.add_artist(legend)
                axsec.legend(loc="lower right", fontsize="xx-large", framealpha=1.)
                axsec.spines["left"].set_visible(False)
                axsec.tick_params(axis="y", labelsize="xx-large", labelcolor="red")
                axsec.set_ylabel(r"Temperature / $\degree$C", size="xx-large", color="red")
            axsec.set_ylim(-20, 15)

        skip_index = find_skip(times)
        offset = 0.01 * (times[-1] - times[0])
        if skip_index != 0:
            ax1.set_xlim(times[0] - offset, times[skip_index] + offset)
            ax2.set_xlim(times[skip_index + 1] - offset, times[-1] + offset)
        else:
            ax1.set_xlim(0, 1)
            ax2.set_xlim(times[0] - offset, times[-1] + offset)
#        ax1.set_xlim(times[0] - 1000000, times[0] + 1000000)
#        ax2.set_xlim(times[0] + 17500000, times[-1] + 100000)
        ax1.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        fig.text(0.5, -0.02, "Date / UTC", size="xx-large", ha="center")
        ax1.set_ylabel("Diff with previous calibration / mV", size="xx-large")


        fig.suptitle(f"Voltage calibration drift, station {station}, {channel_class} channels", size="xx-large")
        fig.tight_layout()

        fig.savefig(figpdf, bbox_inches="tight", format="pdf")
    figpdf.close()
