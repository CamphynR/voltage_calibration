import argparse
import numpy as np
import matplotlib.pyplot as plt

from utility_functions import read_config, unpack_fit_coeff, unpack_fit_residuals, fitted_adc_with_res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = '%(prog)s')
    parser.add_argument("vc")
    parser.add_argument("--channel", type = int, default = 0)
    parser.add_argument("--config", default = "config.json")
    args = parser.parse_args()
    config = read_config(args.config)

    coeff = unpack_fit_coeff(args.vc)
    res_v, res = unpack_fit_residuals(args.vc)

    # the fit range is -1.3, 0.7 but due to the fit sometimes being from -1.299.. to
    # this avoids interpolation crashes
    v_range = np.arange(-1.29, 0.69, 0.05)

    fig, ax = plt.subplots(1, 1, figsize = (18, 12))

    DAC = int(args.channel > 11)
    adc_vc = [fitted_adc_with_res(v_range, coeff[DAC, s], res_v[:, DAC], res[:, DAC]) for s in range(4096)]
    adc_vc_mean = np.mean(adc_vc, axis = 0)
    adc_vc_std = np.std(adc_vc, axis = 0)

    adc_lin = v_range * 4095/2.5

    ax.plot(v_range, adc_lin, label = "linear", color = "red")
    ax.plot(v_range, adc_vc_mean, label = "fit", color = "blue")
    ax.fill_between(v_range, adc_vc_mean - 0.5 * adc_vc_std, adc_vc_mean + 0.5 * adc_vc_std, alpha = 0.5, color = "red")
    color_ped = "black"
    ax.hlines(0, -1.3, 0.7, ls = "dashed", color = color_ped, label = "pedestal")
    ax.vlines(0, -1500, 1000, ls = "dashed", color = color_ped)
    ax.set_xlabel("V")
    ax.set_ylabel("ADC")
    ax.set_xlim(-0.5, 0.3)
    ax.set_ylim(-1200, 1000)
    ax.legend(loc = "best")

    fig.suptitle(f"Test of vc curve, channel {args.channel}")
    figname = f"{config['fig_dir']}/vc_test_curve_ch{args.channel}"
    fig.savefig(figname, bbox_inches = "tight")