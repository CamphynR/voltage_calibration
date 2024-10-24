import argparse
import numpy as np
import matplotlib.pyplot as plt

from utility_functions import read_config, unpack_fit_coeff, unpack_fit_residuals, fitted_adc_with_res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = '%(prog)s')
    parser.add_argument("vc")
    parser.add_argument("--config", default = "config.json")
    args = parser.parse_args()
    config = read_config(args.config)

    coeff = unpack_fit_coeff(args.vc)
    res_v, res = unpack_fit_residuals(args.vc)

    # the fit range is -1.3, 0.7 but due to the fit sometimes being from -1.299.. to
    # this avoids interpolation crashes
    v_range = np.arange(-1.29, 0.69, 0.05)

    fig, axs = plt.subplots(6, 4, figsize = (18, 14), sharey=True, sharex = True)
    axs = np.ndarray.flatten(axs)

    DAC = 0

    for i, ch in enumerate(np.arange(24)):
        if ch == 12:
            DAC = 1
        adc_vc = [fitted_adc_with_res(v_range, coeff[ch, s], res_v[:, DAC], res[:, DAC]) for s in range(4096)]
        adc_vc_mean = np.mean(adc_vc, axis = 0)
        adc_vc_std = np.std(adc_vc, axis = 0)
        axs[i].plot(v_range, adc_vc_mean)
        axs[i].fill_between(v_range, adc_vc_mean - 0.5 * adc_vc_std, adc_vc_mean + 0.5 * adc_vc_std, alpha = 0.5, color = "red")
        axs[i].set_xlabel("V")
        axs[i].set_ylabel("ADC")

    fig.suptitle("Test of vc curves")
    figname = f"{config['fig_dir']}/vc_test_curve"
    fig.savefig(figname, bbox_inches = "tight")