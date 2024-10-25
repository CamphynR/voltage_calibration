import argparse
import numpy as np
import matplotlib.pyplot as plt

from utility_functions import read_config , unpack_vbias, unpack_adc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = '%(prog)s')
    parser.add_argument("bias")
    parser.add_argument("--channel", type = int, default = 0)
    parser.add_argument("--config", default = "config.json")
    args = parser.parse_args()
    config = read_config(args.config)

    vbias = unpack_vbias(args.bias)
    adc = unpack_adc(args.bias)

    fig, ax = plt.subplots(1, 1, figsize = (18, 12))

    DAC = int(args.channel > 11)
    adc_vc_mean = np.mean(adc[:, args.channel], axis = -1)
    adc_vc_std = np.std(adc[:, args.channel], axis = -1)

    adc_lin = vbias[:, DAC] * 4095/2.5

    ax.plot(vbias[:, DAC], adc_lin, label = "linear", color = "red")
    ax.plot(vbias[:, DAC], adc_vc_mean, label = "scan", color = "blue")
    ax.fill_between(vbias[:, DAC], adc_vc_mean - 0.5 * adc_vc_std, adc_vc_mean + 0.5 * adc_vc_std, alpha = 0.5, color = "red")
    color_ped = "black"
    ax.vlines(1.5, 1000, 3000, ls = "dashed", color = color_ped)
    ax.set_xlabel("V")
    ax.set_ylabel("ADC")
    ax.set_xlim(0.2, 2.2)
    ax.legend(loc = "best")

    fig.suptitle(f"Test of bias scan, channel {args.channel}")
    figname = f"{config['fig_dir']}/bias_scan_test_ch{args.channel}"
    fig.savefig(figname, bbox_inches = "tight")