# todo: restructure example: start with basic fmu handler functionality then introduce use case + examples with convinient funcs; Always use TZ FMU"
""""
Goals of this part of the examples:

1. Learn how to perform discrete (stepwise) FMU simulation (class FMU_Discrete)
2. Understand use cases for discrete FMU simulation
3. Set up Experiment using configuration
4. Learn simulating an FMU interacting with python code
5. Learn simulating two (or more) FMUs interacting together
6. Learn different ways to apply inputs (long-term vs. step-specific)
7. Learn how to access the results
8. Advanced: Access basic FMU handler utilities for custom framework
"""

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


# python PID controller
class PID:
    """
    PID implementation from aku and pst, simplified for the needs in this example by kbe
    """
    def __init__(self, Kp=1.0, Ti=100.0, Td=0.0, lim_low=0.0, lim_high=100.0,
                 reverse_act=False, fixed_dt=1.0):

        self.x_act = 0  # measurement
        self.x_set = 0  # set point
        self.e = 0  # control difference
        self.e_last = 0  # control difference of previous time step
        self.y = 0  # controller output
        self.i = 0  # integrator value

        self.Kp = Kp
        self.Ti = Ti
        self.Td = Td
        self.lim_low = lim_low  # low control limit
        self.lim_high = lim_high  # high control limit
        self.reverse_act = reverse_act  # control action
        self.dt = fixed_dt

    # -------- PID algorithm -----------------
    def run(self, x_act, x_set):
        self.x_act = x_act
        self.x_set = x_set

        # control difference depending on control direction
        if self.reverse_act:
            self.e = -(self.x_set - self.x_act)
        else:
            self.e = (self.x_set - self.x_act)

        # Integral
        if self.Ti > 0:
            self.i = 1 / self.Ti * self.e * self.dt + self.i
        else:
            self.i = 0

        # differential
        if self.dt > 0 and self.Td:
            de = self.Td * (self.e - self.e_last) / self.dt
        else:
            de = 0

        # PID output
        self.y = self.Kp * (self.e + self.i + de)

        # Limiter
        if self.y < self.lim_low:
            self.y = self.lim_low
            self.i = self.y / self.Kp - self.e
        elif self.y > self.lim_high:
            self.y = self.lim_high
            self.i = self.y / self.Kp - self.e

        self.e_last = self.e
        return self.y


# plotting format settings
def plotting_fmt():
    """
    Adjusts the plotting format
    """
    # format settings
    import matplotlib
    # plot settings
    matplotlib.rcParams['mathtext.fontset'] = 'custom'
    matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    matplotlib.rcParams['font.size'] = 9
    matplotlib.rcParams['lines.linewidth'] = 0.75


# ######### Use Case of Discrete FMU Simulation ####################

# !!! For co-simulation with more than 2 FMUs consider using AgentLib (E.ON ERC EBC intern) !!!

# Discrete (stepwise) FMU simulation is common for control tasks or co-simulation
# (a simulated model requires feedback as input based on its own output).

# In this example a control task is demonstrated in two scenarios:
# A: System FMU and Python controller
# B: System FMU and controller FMU
#
# Exemplary, a thermal zone FMU model is used as system, that uses a control bus as interface.
# In case B, the controller FMU uses the same interface.
# bus.processVar:         zone temperature measurement
# bus.setPoint:           zone temperature set point
# bus.controlOutput:      relative heating power
# bus.disturbance[1]:     ambient air temperature

# ################# Simulation Setup and Experiment Configuration ###########################
# store simulation setup as dict
start: float = 0  # start time in seconds
stop: float = 86400 * 1  # end time in seconds
output_step: float = 60 * 10  # resolution of simulation results in seconds
comm_step: float = 60 / 3  # step size of FMU communication in seconds.
# In this interval, values are set to or read from the fmu

setup_dict = {
    "start_time": start,
    "stop_time": stop,
    "output_interval": output_step,
    "comm_step_size": comm_step
}

# define working directory (for log file and temporary fmu file extraction)
cd = pathlib.Path(__file__).parent.joinpath("results")

# create experiment setup for system FMU as dict
config_dict = {
    'file_path': pathlib.Path(__file__).parent.joinpath("data", "ThermalZone_bus.fmu"),
    'cd': cd,
    'sim_setup': setup_dict,
    # input data can be passed as .csv file, pd.DataFrame oder TimeSeriesData object
    # 'input_data': pathlib.Path(__file__).parent.joinpath("data", "ThermalZone_input.csv")
}

# ################ Instantiate Simulation API for System FMU ##########################
system = FMU_Discrete(config_dict)

# In the passed configuration dict, the fmu path, the working directory and the simulation setup were set.
# Calling get_experiment_config_fields() reveals that input data can be set too. Input data is passed afterwards
print('Supported configuration options: ', system.get_experiment_config_fields())

# The model inputs are added to 'result_names' by default in addition to the outputs in case of discrete FMU simulation
print("Variables to store when simulating:", system.result_names)

# ################ Create Input Data for the Simulation ###############################
# Without having passed long-term input data with the configuration, message appears in the console during instantiation
# Input data is created in the following and applied using the setter

# The input_table attribute considers input data that holds for a relevant simulation period (here the whole simulation)

# In this example the desired zone temperature and the ambient temperature are known in advance
# and will be set to the input_table attribute

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
# create csv file to access input data later on
# for re-import column naming 'time' is crucial
input_df.to_csv('data/ThermalZone_input.csv', index=True, index_label='time')

# Set the input data to the input_table property
system.input_table = input_df
# alternatively path .csv file path
# system.input_table = pathlib.Path(__file__).parent.joinpath("data", "ThermalZone_input.csv")

# ####################### Initialize System FMU for Discrete Simulation ##########################
# define initial values and parameters
t_start = 293.15 - 5  # parameter
t_start_amb = 293.15 - 15  # initial value

# initialize system FMU
system.initialize_discrete_sim(parameters={'T_start': t_start}, init_values={'bus.disturbance[1]': t_start_amb})
print('Initial results data frame "sim_res": ')
print(system.sim_res)

# ################ A: Simulate System FMU Interacting with python controller ##########################
# Instantiate python PID controller
# Note that the controllers sampling time matches the FMUs communication step size
ctr = PID(Kp=0.01, Ti=300, lim_high=1, reverse_act=False, fixed_dt=comm_step)

# Initialize a running variable for the results of each simulation step
res_step = system.sim_res.iloc[-1].to_dict()

print('Study A: System FMU with Python Controller')
# The do_step function returns True, when stop time is reached and thus breaks the loop
while not system.finished:
    # Call controller
    # (for advanced control strategies that require previous results, use the attribute sim_res and adjust output_interval)
    ctr_action = ctr.run(res_step['bus.processVar'], input_df.loc[system.current_time]['bus.setPoint'])
    # Apply control action to system and perform simulation step
    res_step = system.do_step(input_step={'bus.controlOutput': ctr_action})

# ################# Read Simulation Results ###################################################
# simulation results stored in the attribute 'sim_res' can be returned calling 'get_results()'
results_A = system.get_results()

# ####################### Instantiate and Initialize system and controller FMU #################
# re-initializing the system fmu resets the results (the same instance as before is used)
system.initialize_discrete_sim(parameters={'T_start': t_start}, init_values={'bus.disturbance[1]': t_start_amb})

# A controller FMU is used alternatively to the python controller
# This time the input data is set in the configuration using the generated .csv-file
config_ctr_dict = {
    # compared to the system FMU only the fmu file_path differs
    'file_path': pathlib.Path(__file__).parent.joinpath("data", "PI_1_bus.fmu"),
    'cd': cd,
    'sim_setup': setup_dict,
    'input_data': pathlib.Path(__file__).parent.joinpath("data", "ThermalZone_input.csv")
}

controller = FMU_Discrete(config_ctr_dict)
controller.initialize_discrete_sim()

# ################ B: Simulate System FMU Interacting with a Controller FMU ##########################

res_step = system.sim_res.iloc[-1].to_dict()
print('Study B: System FMU with Controller FMU')
while not system.finished:
    # Call controller and extract control output
    # (for advanced control strategies that require previous results, use the attribute sim_res and adjust output_interval)
    ctr_action = controller.do_step(input_step={'bus.processVar': res_step['bus.processVar']})['bus.controlOutput']
    # write controller output to system FMU as well as pre-known inputs and perform step
    res_step = system.do_step(input_step={'bus.controlOutput': ctr_action})

# read simulation results
results_B = system.get_results()

# ################### Close FMUs ##########################################
# instead of closing each FMU, all FMUs can be closed at once
# # system.close()
# # controller.close()
FMU_Discrete.close_all()

# ###################### Plot Results #########################################
# apply plotting format settings
plotting_fmt()

cases = [results_A, results_B]
time_index_out = np.arange(0, stop + comm_step, output_step)  # time index with output interval step
fig, axes_mat = plt.subplots(nrows=3, ncols=2)
for i in range(len(cases)):
    axes = axes_mat[:, i]
    axes[0].plot(time_index_out, cases[i]['bus.processVar'] - 273.15, label='mea', color='b')
    axes[0].plot(time_index, setpoint - 273.15, label='set', color='r')
    axes[0].set_ylim(15,22)
    axes[1].plot(time_index_out, cases[i]['bus.controlOutput'], label='control output', color='b')
    axes[1].set_ylim(-0.05, 0.2)
    axes[2].plot(time_index_out, cases[i]['bus.disturbance[1]'] - 273.15, label='dist', color='b')
    axes[2].set_ylim(0,40)

    # x label
    axes[2].set_xlabel('Time / s')
    # title and y label
    if i == 0:
        axes[0].set_title('System FMU - Python controller')
        axes[0].set_ylabel('Zone temperature / °C')
        axes[1].set_ylabel('Rel. heating power / -')
        axes[2].set_ylabel('Ambient temperature / °C')
    if i == 1:
        axes[0].set_title('System FMU - Controller FMU')
        axes[0].legend(loc='upper right')
    # grid
    for ax in axes:
        ax.grid(True, 'both')
        if i > 0:
            # ignore y labels for all but the first
            ax.set_yticklabels([])
    for k in range(2):
        axes[k].set_xticklabels([])

plt.tight_layout()
plt.show()

# ###################### Understanding the results ##########################################

# Only a heating device is implemented in the Thermal zone.
# Therefore, the output of the relative heating power is limited to 0.
# Consequently, the controller is unable to cool down the thermal zone.
# This explains most of the control deviation.

# In case you experience oscillating signals, check if the sampling time (communication step size)
# is appropriate for the controller settings

# #################### Advanced: Basic FMU Handler functionality for Customized Application #################
# The FMU_Discrete class includes basic FMU handler utilities previously found in aku's fmu handler skript
# They are demonstrated in the following with a heat pump system fmu

# !!! The _do_step() base function does not append the results to the "sim_res" attribute !!!
# !!! The _do_step() base function does not consider long-term input data (attribute "input_table") !!!

# Instantiate Simulation API for fmu
hp_fmu = FMU_Discrete({'cd': cd,
                       'file_path': pathlib.Path(__file__).parent.joinpath("data", "HeatPumpSystemWithInput.fmu"),
                       'sim_setup': {'stop_time': 6*3600,
                                     'comm_step_size': 10,
                                     'tolerance': 0.0001}
                       })

# define relevant variables
# often relevant quantities are collected in a signal bus in the dymola model
# using a signal bus instance at top level in dymola improves accessability and readability (not the case here).
variables = hp_fmu.find_vars('heatPumpSystem.hPSystemController.sigBusHP')
variables.extend(['TDryBul', 'vol.T', 'senT_a1.T'])

# initialize fmu and set parameter
hp_fmu.initialize_discrete_sim(parameters={'heaCap.C': 100000}, init_values={'TDryBul': 0+273.15})

# initialize stop indicator and list for results
finished = False
results_list = []

# simulation loop
# the ambient temperature is altered during the simulation

# Note that for simulating the fmu with altering inputs that are known in advance
# the simulation api for continuous fmu simulation is a better choice.

while not hp_fmu.finished:
    # read fmu state
    res = hp_fmu.read_variables(variables)
    res.update({'SimTime': hp_fmu.current_time})
    results_list.append(res)
    # set ambient temperature depending on time
    if hp_fmu.current_time <= 3*3600:
        hp_fmu.set_variables({'TDryBul': 0 + 273.15})
    else:
        hp_fmu.set_variables({'TDryBul': 5 + 273.15})
    # perform simulation step
    hp_fmu.step_only()

# close fmu
hp_fmu.close()

# convert list of dicts to pandas datraframe
sim_res_frame = pd.DataFrame(results_list)
sim_res_frame.index = sim_res_frame['SimTime']

# plot electric power depending on ambient temperature
x_values = sim_res_frame['SimTime']
fig, axes = plt.subplots(nrows=4, ncols=1)
axes[0].set_title('Heating curve controlled water-water heat pump')
axes[0].plot(x_values, sim_res_frame['heatPumpSystem.hPSystemController.sigBusHP.PelMea'])
axes[0].set_ylabel('Electric power / W')
axes[1].plot(x_values, sim_res_frame['TDryBul']-273.15)
axes[1].set_ylabel('Ambient T / °C')
axes[2].plot(x_values, sim_res_frame['senT_a1.T']-273.15)
axes[2].set_ylabel('Supply T / °C')
axes[3].plot(x_values, sim_res_frame['vol.T']-273.15)
axes[3].set_ylabel('Room air T / °C')
axes[3].set_xlabel('Time / s')
for i in range(3):
    axes[i].set_xticklabels([])
for ax in axes:
    ax.grid(True, 'both')
plt.tight_layout()
plt.show()
















