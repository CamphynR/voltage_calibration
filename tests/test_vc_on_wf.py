import argparse
import glob
import logging
import numpy as np
import matplotlib.pyplot as plt

from NuRadioReco.modules.io.RNO_G.readRNOGDataMattak import readRNOGData
from NuRadioReco.utilities import units
from utility_functions import read_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = '%(prog)s')
    parser.add_argument("vc")
    parser.add_argument("--run", default = 300)
    parser.add_argument("--station", default = 13)
    parser.add_argument("--backend", default = "pyroot")
    parser.add_argument("--config", default = "config.json")
    args = parser.parse_args()

    config = read_config(args.config)

    run_files = glob.glob(f"{config['data_dir']}/station{args.station}/run{args.run}")

    rnog_reader = readRNOGData(log_level=logging.DEBUG)
    rnog_reader.begin(run_files,
		      # only forced triggers to make sure test doesn't break blinding
		      select_triggers = "FORCE",
                      read_calibrated_data = True,
                      convert_to_voltage=False,
		      apply_baseline_correction = "approximate",
              mattak_kwargs=dict(backend = args.backend,
                                 voltage_calibration = args.vc
		                        )
                     )

    fig, axs = plt.subplots(6, 4, figsize = (18, 14))
    axs = np.ndarray.flatten(axs)

    for event in rnog_reader.run():
        station_id = event.get_station_ids()[0]
        station = event.get_station(station_id)
        for i, channel in enumerate(station.iter_channels()):
            time = channel.get_times()
            wf = channel.get_trace()
            axs[i].plot(time, wf)
        break

    fig.suptitle("Test of calibrated waveforms")
    figname = f"{config['fig_dir']}/vc_test_wf"
    print(f"Saving as {figname}")
    fig.savefig(figname, bbox_inches = "tight")