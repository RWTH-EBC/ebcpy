"""
Goals of this part of the examples:

1. Access basic FMU handler utilities
2. Learn how to perform discrete (stepwise) FMU simulation
3. Understand use cases for discrete FMU simulation
4. Set up Experiment using configuration
5. Learn simulating an FMU interacting with python code
6. Learn simulating two (or more) FMUs interacting together
7. Learn different ways to apply inputs (long-term vs. step-specific)
8. Learn how to access the results
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
        PID controlleR
        :param kp:
            Gain
        :param ti:
            Integral Time Constant
        :param td:
            Derivative Time Constant
        :param lim_low:
            Lower Limit
        :param lim_high:
            Upper Limit
        :param reverse_act:
            For True, the output decreases with an increasing control difference
        :param fixed_dt:
            Fixed sampling rate
        """

    def __init__(self, kp=1.0, ti=100.0, td=0.0, lim_low=0.0, lim_high=100.0,
                 reverse_act=False, fixed_dt=1.0):

        self.x_act = 0  # measurement
        self.x_set = 0  # set point
        self.e = 0  # control difference
        self.e_last = 0  # control difference of previous time step
        self.y = 0  # controller output
        self.i = 0  # integrator value

        self.Kp = kp
        self.Ti = ti
        self.Td = td
        self.lim_low = lim_low  # low control limit
        self.lim_high = lim_high  # high control limit
        self.reverse_act = reverse_act  # control action
        self.dt = fixed_dt

    # -------- PID algorithm -----------------
    def run(self, x_act, x_set):
        """
        Control method, returns control action based on actual value and set point
        :param x_act:
            Measurement
        :param x_set:
            Set point
        :return:
            Control action

        """
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


def main(
        n_days: int = 1,
        log_fmu: bool = True,
        with_plot: bool = True
):
    """
    Arguments of this example:
    :param float n_days:
        Duration of the simulation in days
    :param bool log_fmu:
        Whether to get the FMU log output
    :param bool with_plot:
        Show the plot at the end of the script. Default is True.
    """

    # ########## FMU File and Working Directory ##########################
    # A thermal zone FMU model is used in this example.

    # define working directory (for log file and temporary fmu file extraction)
    cd = pathlib.Path(__file__).parent.joinpath("results")

    # define fmu file path
    file_path = pathlib.Path(__file__).parent.joinpath("data", "ThermalZone_bus.fmu")

    # ######### Basic FMU Handler Utilities ###########################
    # The FMU_Discrete class includes basic FMU handler utilities
    # previously found in aku's fmu handler skript (E.ON ERC EBC intern)

    # Instantiate simulation api with config; fmu file path and working directory are compulsory
    config_dict = {'cd': cd,
                   'file_path': file_path}

    tz_fmu = FMU_Discrete(config_dict)

    # define relevant vars_of_interest
    # often relevant quantities can be found in a signal bus in the dymola model
    # using a signal bus instance at top level in dymola improves accessability and readability
    # in this case the relevant signal bus instance is named "bus".
    vars_of_interest = tz_fmu.find_vars(start_str='bus')
    print('Variables of interest: ', vars_of_interest)
    print('Inputs: ', tz_fmu.inputs)
    print('Outputs: ', tz_fmu.outputs)

    # The investigated thermal zone model uses the signal bus as control interface.
    # It contains the in- and outputs.
    # bus.processVar:         zone temperature measurement
    # bus.controlOutput:      relative heating power
    # bus.disturbance[1]:     ambient air temperature

    # ############## Stepwise FMU Simulation #####################################

    # To simulate the fmu, a simulation setup configuration is required
    step_size = 3600
    setup_dict = {
            "start_time": 0,
            "stop_time": 86400,  # 1 day
            # Simulation steps of 10 min after which variables can be read or set
            "comm_step_size": step_size
    }
    tz_fmu.set_sim_setup(setup_dict)

    # Initialize fmu and set parameters and initial values
    t_start = 20+273.15  # parameter
    t_start_amb = -6+273.15  # initial value
    tz_fmu.initialize_discrete_sim(parameters={'T_start': t_start},
                                   init_values={'bus.disturbance[1]': t_start_amb})

    # Initialize list for results
    # by reading the values of the relevant vars_of_interest from the fmu
    result_list = [tz_fmu.read_variables(vars_of_interest)]

    # simulation loop: Simulation the fmu stepwise for 12 hours and read results every step
    while tz_fmu.current_time < 12 * 3600:
        # perform simulation step
        tz_fmu.step_only()
        # read results and append to list
        res = tz_fmu.read_variables(vars_of_interest)
        result_list.append(res)
        print('Temperature: {}°C'.format(round(res['bus.processVar']-273.15)))

    # After 12 hours, the temperature reaches 18°C.
    # To turn on the heating the according variable is set
    tz_fmu.set_variables({'bus.controlOutput': 0.1})

    # The simulation is continued until the stop time is reached
    while not tz_fmu.finished:
        tz_fmu.step_only()
        result_list.append(tz_fmu.read_variables(vars_of_interest))

    # close fmu
    tz_fmu.close()

    # convert list of dicts to pandas datraframe
    sim_res_frame = pd.DataFrame(result_list)
    sim_res_frame.index = sim_res_frame['SimTime']

    # ########### Plotting ##########################################

    # Plotting the room temperature reveals
    # that turning on the heating could increase the temperature again
    plotting_fmt()  # apply plotting format settings
    x_values = sim_res_frame['SimTime']
    fig, axes = plt.subplots(nrows=3, ncols=1)
    axes[0].set_title('Thermal zone with ideal heating')
    axes[0].plot(x_values, sim_res_frame['bus.processVar']-273.15, color='b')
    axes[0].hlines(18, x_values.iloc[0], x_values.iloc[-1], ls='--', color='r')
    axes[0].set_ylabel('T Zone / °C')
    axes[1].step(x_values, sim_res_frame['bus.controlOutput'], color='b')
    axes[1].set_ylabel('Rel. Power / -')
    axes[2].plot(x_values, sim_res_frame['bus.disturbance[1]']-273.15, color='b')
    axes[2].set_ylabel('T Ambient / °C')

    for i in range(2):
        axes[i].set_xticklabels([])
    for ax in axes:
        ax.grid(True, 'both')
    plt.tight_layout()
    if with_plot:
        plt.show()

    # ######### Use Case of Discrete FMU Simulation ####################

    # The previous investigation is a very simple control task:
    # Because the room temperature gets too low, the heating is activated manually

    # Discrete (stepwise) FMU simulation is common for control tasks or co-simulation
    # (a simulated model requires feedback as input based on its own output).

    # !!! For co-simulation with more than 2 FMUs consider using AgentLib (E.ON ERC EBC intern) !!!

    # In the following, a control task with a PI heating controller
    # is demonstrated in two scenarios:
    # A: System FMU and Python controller
    # B: System FMU and controller FMU

    # # ################# Simulation Setup and Experiment Configuration ###########################
    start: float = 0  # start time in seconds
    stop: float = 86400 * n_days  # end time in seconds
    output_step: float = 60 * 10  # resolution of simulation results in seconds
    comm_step: float = 60 / 3  # step size of FMU communication in seconds
    # In this interval, values are set to or read from the fmu

    # find out supported experiment configuration options
    print(f"Supported experiment configuration: {FMU_Discrete.get_experiment_config_fields()}")
    # find out supported simulation setup options
    print(f"Supported simulation setup: {FMU_Discrete.get_simulation_setup_fields()}")

    # collect simulation setup
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
    input_df = pd.DataFrame({'bus.disturbance[1]': dist, 'bus.setPoint': setpoint},
                            index=time_index)
    # create csv file to access input data later on
    # for re-import column naming 'time' is crucial
    input_df.to_csv('data/ThermalZone_input.csv', index=True, index_label='time')

    # Set the input data to the input_table property
    system.input_table = input_df
    # alternatively pass .csv file path
    # system.input_table = pathlib.Path(__file__).parent.joinpath("data", "ThermalZone_input.csv")

    # ############# Initialize System FMU for Discrete Simulation #######################
    # define initial values and parameters
    t_start = 15 + 273.15  # parameter
    t_start_amb = 5 + 273.15  # initial value

    # initialize system FMU
    system.initialize_discrete_sim(parameters={'T_start': t_start},
                                   init_values={'bus.disturbance[1]': t_start_amb})
    print('Initial results data frame "sim_res_df": ')
    print(system.sim_res_df)

    # ############## A: Simulate System FMU Interacting with python controller #################
    # Instantiate python PID controller
    # Note that the controllers sampling time matches the FMUs communication step size
    ctr = PID(kp=0.01, ti=300, lim_high=1, reverse_act=False, fixed_dt=comm_step)

    # Initialize a running variable for the results of each simulation step
    res_step = system.sim_res_df.iloc[-1].to_dict()

    print('Study A: System FMU with Python Controller')

    # ############ Do Step Function with Extended Functionality ###########################
    # In discrete simulation a simulation step typically goes hand in hand wih
    # setting values to the fmu and reading from the fmu.
    # This is all covered by the do_step() function.
    # It also considers inputs from the input_table attribute.
    # The results are stored in the sim_res_df attribute
    # and cover the variables within the result_names attribute

    while not system.finished:
        # Call controller
        # (for advanced control strategies that require previous results,
        # use the attribute sim_res_df and adjust output_interval)
        ctr_action = ctr.run(
            res_step['bus.processVar'], input_df.loc[system.current_time]['bus.setPoint'])
        # Apply control action to system and perform simulation step
        res_step = system.do_step(input_step={'bus.controlOutput': ctr_action})

    # ################# Read Simulation Results ###################################################
    # simulation results stored in the attribute 'sim_res_df'
    # can be returned calling 'get_results()'
    results_a = system.get_results()

    # ####################### Instantiate and Initialize system and controller FMU ################
    # re-initializing the system fmu resets the results (the same instance as before is used)
    system.initialize_discrete_sim(parameters={'T_start': t_start},
                                   init_values={'bus.disturbance[1]': t_start_amb})

    # A controller FMU is used alternatively to the python controller
    # This time the input data is set in the configuration using the generated .csv-file
    config_ctr_dict = {
        # compared to the system FMU only the fmu file_path differs
        'file_path': pathlib.Path(__file__).parent.joinpath("data", "PI_1_bus.fmu"),
        'cd': cd,
        'sim_setup': setup_dict,
        'input_data': pathlib.Path(__file__).parent.joinpath("data", "ThermalZone_input.csv")
        # input data can be passed as .csv file, pd.DataFrame oder TimeSeriesData object

    }

    controller = FMU_Discrete(config_ctr_dict, log_fmu=log_fmu)
    controller.initialize_discrete_sim()

    # ############# B: Simulate System FMU Interacting with a Controller FMU ##################

    res_step = system.sim_res_df.iloc[-1].to_dict()
    print('Study B: System FMU with Controller FMU')
    while not system.finished:
        # Call controller and extract control output
        # (for advanced control strategies that require previous results,
        # use the attribute sim_res_df and adjust output_interval)
        ctr_action = controller.do_step(
            input_step={'bus.processVar': res_step['bus.processVar']})['bus.controlOutput']
        # write controller output to system FMU as well as pre-known inputs and perform step
        res_step = system.do_step(input_step={'bus.controlOutput': ctr_action})

    # read simulation results
    results_b = system.get_results()

    # ################### Close FMUs ##########################################
    # instead of closing each FMU, all FMUs can be closed at once
    # # system.close()
    # # controller.close()
    FMU_Discrete.close_all()

    # ###################### Plot Results #########################################
    cases = [results_a, results_b]
    # time index with output interval step
    time_index_out = np.arange(0, stop + comm_step, output_step)
    fig, axes_mat = plt.subplots(nrows=3, ncols=2)
    for i in range(len(cases)):
        axes = axes_mat[:, i]
        axes[0].plot(time_index_out, cases[i]['bus.processVar'] - 273.15, label='mea', color='b')
        axes[0].plot(time_index, setpoint - 273.15, label='set', color='r')
        axes[0].set_ylim(15, 22)
        axes[1].plot(time_index_out, cases[i]['bus.controlOutput'],
                     label='control output', color='b')
        axes[1].set_ylim(-0.05, 0.2)
        axes[2].plot(time_index_out, cases[i]['bus.disturbance[1]'] - 273.15,
                     label='dist', color='b')
        axes[2].set_ylim(0, 40)

        # x label
        axes[2].set_xlabel('Time / s')
        # title and y label
        if i == 0:
            axes[0].set_title('System FMU - Python controller')
            axes[0].set_ylabel('T Zone / °C')
            axes[1].set_ylabel('Rel. Power / -')
            axes[2].set_ylabel('T Amb / °C')
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
    if with_plot:
        plt.show()

    # ###################### Understanding the results ##########################################

    # Only a heating device is implemented in the Thermal zone.
    # Therefore, the output of the relative heating power is limited to 0.
    # Consequently, the controller is unable to cool down the thermal zone.
    # This explains most of the control deviation.

    # In case you experience oscillating signals, check if the sampling time
    # (communication step size) is appropriate for the controller settings


if __name__ == '__main__':
    main(
        log_fmu=True,
        with_plot=True,
        n_days=1
    )
