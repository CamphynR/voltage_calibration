import argparse
import glob
import time
import numpy as np

from voltageCalibration import voltageCalibration
from utility_functions import read_config, save_to_pickle


def get_vc_diff(vc_0, vc_1):
    return np.mean(np.abs(vc_1 - vc_0), axis = -1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = '%(prog)s')
    parser.add_argument("-s", "--station", default=13)
    parser.add_argument("--vc_ref_file", default = None)
    parser.add_argument("--config", default = "config.json")
    parser.add_argument("--exclude_2022", action = "store_true")
    parser.add_argument("--no_window", action="store_true")
    args = parser.parse_args()

    config = read_config(args.config)

    samples = list(range(4096))
    channels = list(range(24))

    vc_files = glob.glob(f"{config['bias_dir']}/station{args.station}/run*/vol*")
    if not args.exclude_2022:
        vc_files += glob.glob(f"{config['bias_dir']}/season22_scan_testing/station{args.station}/vol*s{args.station}*")

    vc_files = sorted(vc_files, key = lambda filename : filename.split("/")[-1])
    if args.vc_ref_file is None:
        args.vc_ref_file = vc_files[0]
    
    vcRef = voltageCalibration(args.vc_ref_file)
    if args.no_window:
        v_range = np.arange(-1.2, 0.7, 0.1)
    else:
        step = 0.02
        v_range = np.arange(config['window'][0], config['window'][1] + step, step)
    
    t_ref = vcRef.get_times()[0]
    adc_ref = np.array(
                [[vcRef.get_fit_curve(v_range, ch, s) for s in samples] for ch in channels])

    t = []
    adc_diffs = []
    for i, vc_file in enumerate(vc_files):
        print(f"vc {i}/{len(vc_files) - 1}")
        vc = voltageCalibration(vc_file)
        t.append(vc.get_times()[0])
        adc = np.array(
                [[vc.get_fit_curve(v_range, ch, s) for s in samples] for ch in channels])
        adc_diff = get_vc_diff(adc_ref, adc)
        adc_diffs.append(adc_diff)
    
    pickle_dict = dict(times = t, adc_diffs = adc_diffs, station = args.station, v = v_range,
                       vc_ref_file = args.vc_ref_file, window = config['window'])
    pickle_name = f"{config['pickle_dir']}/vc_drift_ref/vc_drift_ref_s{args.station}"
    if not args.no_window:
        pickle_name += f"_w{config['window'][0]:.1f}_{config['window'][1]:.1f}"
    else:
        pickle_name += "_no_window"
    save_to_pickle(pickle_dict, pickle_name)