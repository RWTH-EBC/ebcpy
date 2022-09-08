import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
# Imports from ebcpy
from ebcpy import FMU_Discrete
# import for python controller
import time

n_days=1
log_fmu = False


# define working directory (for log file and temporary fmu file extraction)
cd = pathlib.Path(__file__).parent.joinpath("results")

# define fmu file path
file_path = pathlib.Path(__file__).parent.joinpath("data", "ThermalZone_bus.fmu")

start: float = 0  # start time in seconds
stop: float = 86400 * n_days  # end time in seconds
output_step: float = 60 * 10  # resolution of simulation results in seconds
comm_step: float = 60 / 3  # step size of FMU communication in seconds

setup_dict = {
        "start_time": start,
        "stop_time": stop,
        "output_interval": output_step,
        "comm_step_size": comm_step
    }

# create experiment configuration for system FMU as dict
config_dict = {
    'file_path': file_path,
    'cd': cd,
    'sim_setup': setup_dict,
}

# ################ Instantiate Simulation API for System FMU ##########################
system = FMU_Discrete(config_dict, log_fmu=log_fmu)
# A warning shows that no long-term input data has been set yet. It will be set later.

# The model inputs are added to 'result_names' by default in addition to the outputs
# in case of discrete FMU simulation
print("Variables to store when simulating:", system.result_names)

# ################ Create Input Data for the Simulation ###############################
# Without having passed long-term input data with the configuration,
# a message appears in the console during instantiation
# Input data is created in the following and applied using the setter

# The input_table attribute considers input data that holds for a relevant simulation period
# (here for the whole simulation)

# In this example the desired zone temperature and the ambient temperature are known in advance
# and will be set to the input_table attribute

time_index = np.arange(start, stop + comm_step, comm_step)
# The ambient air temperature (bus.disturbance[1]) is modeled as cosine function
dist = 293.15 - np.cos(time_index/86400*2*np.pi) * 15
# The desired zone temperature (bus.setPoint) considers a night setback:
setpoint = np.ones(len(time_index)) * 290.15
for idx in range(len(time_index)):
    sec_of_day = time_index[idx] - math.floor(time_index[idx]/86400) * 86400
    if 3600 * 8 < sec_of_day < 3600 * 17:
        setpoint[idx] = 293.15
# Store input data as pandas DataFrame
input_df = pd.DataFrame({'bus.disturbance[1]': dist, 'bus.setPoint': setpoint}, index=time_index)
# create csv file to access input data later on
# for re-import column naming 'time' is crucial
input_df.to_csv('data/ThermalZone_input.csv', index=True, index_label='time')

# Set the input data to the input_table property
system.input_table = input_df

sim_setup_idx = np.arange(system.sim_setup.start_time,
                          system.sim_setup.stop_time + system.sim_setup.comm_step_size,
                          system.sim_setup.comm_step_size).tolist()

table_idx = system.input_table.index.tolist()

system.close()