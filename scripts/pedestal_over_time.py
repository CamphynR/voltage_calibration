import argparse
import glob
import json
import uproot
import numpy as np
import matplotlib.pyplot as plt

from utility_functions import save_to_pickle
from biasParser import biasParser
        

def read_time_and_pedestal(run_path):
    pedestal_value = None
    for pedestal_name in ["pedestal.root", "pedestals.root"]:
        try:
            with uproot.open(f"{run_path}/{pedestal_name}") as pedestal:
                pedestal_time = pedestal["pedestals"]["when"].array(library = "np")
                pedestal_value = pedestal["pedestals"]["vbias[2]"].array(library = "np")
            break
        except:
            continue
    if pedestal_value is None:
        return None
    return [pedestal_time, pedestal_value]



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = '%(prog)s')
    parser.add_argument("--station", default=13)
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()


    with open(args.config, "r") as config_json:
        config = json.load(config_json)

    biasParser = biasParser(f"{config['bias_dir']}/station{args.station}")
    biasParser.set_function(read_time_and_pedestal)
    pedestals = biasParser.run()
    pedestal_times = [ped[0] for ped in pedestals]
    pedestals = [ped[1] for ped in pedestals]

    ped_dict = dict(pedestal_times = pedestal_times, pedestals = pedestals, station = args.station)
    pickle_path = f"{config['pickle_dir']}/pedestals/ped_over_time_s{args.station}.pickle"
    save_to_pickle(ped_dict, pickle_path)