# Start by importing all relevant packages
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
# Imports from ebcpy
from ebcpy import FMU_Discrete
# import for python controller
import time

config_dict = {
        'file_path': pathlib.Path(__file__).parent.joinpath("data", "ThermalZone_bus.fmu"),
        }
fmu = FMU_Discrete(config_dict)

fmu.sim_setup

start: float = 0  # start time in seconds
stop: float = 86400 * 1  # end time in seconds
output_step: float = 60 * 10  # resolution of simulation results in seconds
comm_step: float = 60 / 3  # step size of FMU communication in seconds.

time_index = np.arange(start, stop + comm_step, comm_step)
# The ambient air temperature (bus.disturbance[1]) is modeled as cosine function
dist = 293.15 - np.cos(time_index/86400*2*np.pi) * 15
# The desired zone temperature (bus.setPoint) considers a night setback:
setpoint = np.ones(len(time_index)) * 290.15
for idx in range(len(time_index)):
        sec_of_day = time_index[idx] - math.floor(time_index[idx]/86400) * 86400
        if sec_of_day > 3600 * 8 and sec_of_day < 3600 * 17:
            setpoint[idx] = 293.15
# Store input data as pandas DataFrame
input_df = pd.DataFrame({'bus.disturbance[1]': dist, 'bus.setPoint': setpoint}, index=time_index)


fmu.input_table =  pathlib.Path(__file__).parent.joinpath("data", "fake_inp.txt")
