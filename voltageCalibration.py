"""
This file contains all classes used in the voltage calibration evaluation
"""

import json
import re
import time

import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import uproot



@jit(nopython = True)
def make_vc_curve(v, coeff, vres, res):
    """
    Function for 1 channel, 1 sample
    Coeff sequence assumed to be [0th order, 1th order, 2nd order, ..]
    """
    poly_eval = np.zeros(len(v), dtype = np.float32)
    for i, v in enumerate(v):
        for j, r_v in enumerate(vres):
            if v < r_v:
                idx = j
                break
        res_inter = ((res[idx] - res[idx - 1]) / (vres[idx] - vres[idx - 1])) * (v - vres[idx - 1]) + res[idx - 1]
        var_list = np.array([v**i for i in range(10)], dtype = np.float32)
        evaluated = coeff.dot(var_list)
        poly_eval[i] = evaluated + res_inter
    return poly_eval


class voltageCalibration():
    """
    Class to read in voltage calibration constants and residuals from VC*.root files
    """
    def __init__(self, path) -> None:

        with uproot.open(path) as f:
            coeffs = np.stack(f["coeffs_tree/coeff"].array(library = 'np'))
            start_time = f["general_tree/startTime"].array(library = "np")[0]
            end_time = f["general_tree/endTime"].array(library = "np")[0]
            vres_dac1, vres_dac2 = f["aveResid_dac1"].member("fX"), f["aveResid_dac2"].member("fX")
            residual_dac1, residual_dac2 = f["aveResid_dac1"].member("fY"), f["aveResid_dac2"].member("fY")
            diff = len(vres_dac1) - len(vres_dac2)
            if diff > 0:
                vres_dac1 = vres_dac1[: -diff]
                residual_dac1 = residual_dac1[: -diff]
            elif diff < 0:
                vres_dac2 = vres_dac2[:diff]
                residual_dac2 = residual_dac2[: diff]
        
        self.coeffs = coeffs.reshape((24, 4096,10))
        self.times = start_time, end_time
        self.vres = np.stack(np.array([vres_dac1, vres_dac2]), axis = -1)
        self.res = np.stack(np.array([residual_dac1, residual_dac2]), axis = -1)

    def get_fit_curve(self, v, channel, sample):
        """
        v : shape [points, DAC]
        """
        dac = int(channel > 11)
        adc = make_vc_curve(v, self.coeffs[channel, sample], self.vres[:, dac], self.res[:, dac])

        return adc

    def get_times(self):
        return self.times