import pathlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
# Imports from ebcpy
from ebcpy import FMU_API
# import for controller
import time

"""
Demonstration of stepwise FMU simulation.
Stepwise FMU simulation is common for control tasks or co-simulation
(the simulated model requires feedback as input based on its on output).

In this example a control task is demonstrated in two scenarios: 
A: System FMU and Python controller
B: System FMU and controller FMU

Exemplary, a thermal zone FMU model is used as system, that uses a control bus as interface. 
In case B, the controller FMU uses the same interface. 
bus.processVar:         zone temperature measurement
bus.setPoint:           zone temperature set point
bus.controlOutput:      relative heating power
bus.disturbance[1]:     ambient air temperature

During the stepwise simulation, two different input types can be applied onto an FMU: 
An input table with preassigned values (in this case bus.setPoint and bus.disturbance[1]) 
and input that is only valid for one step (bus.controlOutput, and in case of the controller FMU bus.processVar).

For co-simulation with more than 2 FMUs consider the use of AgentLib
"""


class PID:
    '''
    PID implementation from aku and pst, simplified for the needs in this example by kbe
    '''
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


# ------ Settings, input data, initial values, system FMU ---------------------------

# ---- Settings ---------
output_step = 60*10  # step size of simulation results in seconds (resolution of results data)
comm_step = 60/3  # step size of FMU communication in seconds (in this interval, values are set to or read from the fmu)
start = 0  # start time
stop = 86400 * 0.25  # stop time
# store simulation setup as dict  # fixme: add comm_step??
simulation_setup = {"start_time": start, "stop_time": stop, "output_interval": output_step}

# ---- Input data table--------
# Input data, that is known in advance will be passed as an input data table:
time_index = np.arange(start, stop + comm_step, comm_step)
# The ambient air temperature (bus.disturbance[1]) is modeled as sine function
dist = 293.15 - np.cos(time_index/86400*2*np.pi) * 15
# The desired zone temperature considers a night setback:
setpoint = np.ones(len(time_index)) * 290.15
for idx in range(len(time_index)):
    sec_of_day = time_index[idx] - math.floor(time_index[idx]/86400) * 86400
    if sec_of_day > 3600 * 8 and sec_of_day < 3600 * 17:
        setpoint[idx] = 293.15
# Store input data as pandas DataFrame
input_data = pd.DataFrame({'bus.disturbance[1]': dist, 'bus.setPoint': setpoint}, index=time_index)  # todo: print warning that only the variables that match with fmu vars are set

# --- Initial values and parameters ------
t_start = 293.15 - 5
t_start_amb = 293.15 - 15

# ---- System FMU ---- todo: move settings to FMU class and config file maybe
work_dir = pathlib.Path(__file__).parent.joinpath("results")
# path to fmu file  # todo: move settings to FMU class and config file maybe
path = pathlib.Path(__file__).parent.joinpath("data", "ThermalZone_bus.fmu")
# create FMU API object
sys = FMU_API(model_name=path, cd=work_dir, input_data=input_data, n_cpu=1, log_fmu=False)  # Todo: allow path as expected type for model_name additionaly
# set custom simulation setup
sys.set_sim_setup(sim_setup=simulation_setup)  # Todo: Changed to property function in v0.1.7??


# ----- Study A: System FMU - Python controller ---------------------------------------------

# --- Initialize system FMU ------
sys.initialize_fmu_for_do_step(parameters={'T_start': t_start},
                               init_values={'bus.disturbance[1]': t_start_amb},  # fixme: does this impact the simulation or the output only??
                               css=comm_step,  # communication step size
                               tolerance=None,  # preset value will be used
                               store_input=True)  # default; the FMU inputs are added to the simulation results

# By default, the FMU in- and outputs are also added to the list of variables to read
print("Variables to store when simulating:", sys.result_names)

# ---- Initialize controller ----------
ctr = PID(Kp=0.01, Ti=300, lim_high=1, reverse_act=False, fixed_dt=60)  # todo: why not possible to use other PID notation? Problem then with kp<1

# ------ Simulation Loop -------
# The central FMU functions used in the loop are read_variables_wr() and set_variables_wr()
# These functions extend the functionality of the basic FMU communication functions read_variables() and set_variables()
# 1. read_variables_wr reads and the values for the variables stored in fmu_api.resultnames.
# It further appends the results to the fmu_api.sim_res attribute
# 2. set_variables_wr writes the values in the input data table and inputs for the specific step to the FMU.
# Values from the input data table are either hold or interpolated.
# By default, a simulation step is performed after writing. Optionally, the FMU can be closed when stop time is reached.
while not sys.finished:
    # Read system state from FMU
    res_step = sys.read_variables_wr(save_results=True)  # default; the results are appended to fmu_api.sim_res
    # Call controller (for advanced control strategies that require previous results, use the attribute sim_res)
    ctr_action = ctr.run(res_step['bus.processVar'], input_data.loc[sys.current_time]['bus.setPoint'])#.values[0])
    # write controller output to system FMU as well as pre-known inputs and perform step
    sys.set_variables_wr(input_step={'bus.controlOutput': ctr_action},
                         do_step=True  # default; a simulation step is performed after writing
                    )  # default; the FMU is not closed when finished for second study

# ---- Results ---------
# return simulation results as pd data frame
results_study_A = sys.get_results(tsd_format=False)  # optional; the results are returned as pandas DataFrame


# ----- Study B: System FMU - Controller FMU ---------------------------------------------

# --- Initialize system FMU ------
# reinitialization resets the results
sys.initialize_fmu_for_do_step(parameters={'T_start': t_start},
                                   init_values={'bus.disturbance[1]': t_start_amb},  # fixme: does this impact the simulation or the output only??
                                   css=comm_step,  # communication step size
                                   tolerance=None,  # preset value will be used
                                   store_input=True)  # default; the FMU inputs are added to the simulation results

# ------ Instantiate and initialize controller FMU-----
# path of fmu file  # todo: move settings to FMU class and config file maybe
path = pathlib.Path(__file__).parent.joinpath("data", "PI_1_bus.fmu")
ctr = FMU_API(model_name=path, cd=work_dir, input_data=input_data, log_fmu=False)  # Todo: allow path as expected type for model_name additionaly
ctr.set_sim_setup(sim_setup=simulation_setup)
ctr.initialize_fmu_for_do_step(parameters=None,  # Not required for controller
                               init_values=None,  # not required for controller
                               css=comm_step,  # communication step size
                               tolerance=None,  # preset value will be used
                               store_input=False)  # optional; the FMU inputs are not added to the results

# ------ Simulation Loop -------
while not sys.finished:
    # read system state from system FMU
    res_step = sys.read_variables_wr()
    # call controller (for advanced control strategies that require previous results, use the attribute sim_res)
    ctr.set_variables_wr(input_step= {'bus.processVar': res_step['bus.processVar']})
    ctr_action = ctr.read_variables_wr(save_results=False)['bus.controlOutput']  # results of controller are not saved
    # write controller output to system FMU as well as pre-known inputs and perform step
    sys.set_variables_wr(input_step={'bus.controlOutput': ctr_action})  # optional; fmu is closed as not needed anymore

# sys.close()
# ctr.close()
FMU_API.close_all()

# return simulation results as pd data frame
results_study_B = sys.get_results(tsd_format=False)


# -------------------- Plotting -------------------------------------------------

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

cases = [results_study_A, results_study_B]
time_index_out = np.arange(0, stop + comm_step, output_step)  # time index with output interval step
fig, axes_mat = plt.subplots(nrows=3, ncols=2)
for i in range(len(cases)):
    axes = axes_mat[:, i]
    axes[0].plot(time_index_out, cases[i]['bus.processVar'] - 273.15, label='mea', color='b')
    axes[0].plot(time_index, setpoint - 273.15, label='set', color='r')  # fixme: setpoint not available in results
    axes[1].plot(time_index_out, cases[i]['bus.controlOutput'], label='control output', color='b')
    axes[2].plot(time_index_out, cases[i]['bus.disturbance[1]'] - 273.15, label='dist', color='b')

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

plt.tight_layout()
plt.show()

# Comment on the results:

# Only a heating device is implemented in the Thermal zone.
# Therefore, the output of the relative heating power is limited to 0.
# Consequently, the controller is unable to cool down the thermal zone.
# This explains most of the control deviation.

# In case you experience oscillating signals, check if the sampling time (communication step size)
# is appropriate for the controller settings

















