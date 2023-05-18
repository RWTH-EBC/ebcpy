"""
Goals of this part of the examples:

1. Learn how to perform discrete (stepwise) FMU simulation
"""

from ebcpy import FMU_API_Dis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main(
        n_days: int = 1,
        with_plot: bool = True
):
    """
    Arguments of this example:
    :param float n_days:
        Duration of the simulation in days
    :param bool with_plot:
        Show the plot at the end of the script. Default is True.
    """

    # ################## FMU File and Use Case ##########################
    # A (Dymola-exported) thermal zone FMU model is used in this example.

    # define start values
    t_start = 20 + 273.15  # parameter
    t_start_amb = -6 + 273.15  # initial value
    start_vals = {'T_start': t_start,
                  'bus.disturbance[1]': t_start_amb}

    # create fmu object
    tz_fmu = FMU_API_Dis(stop_time=24*3600,  # 1 day
                         step_size=1*3600,  # 1 hour
                         fmu_file='data/ThermalZone_bus.fmu',
                         start_values=start_vals
                         )

    # find variables of interest for results
    # often relevant quantities can be found in a signal bus (expandable connector) in the dymola model
    # using a signal bus instance at top level in dymola improves accessability and readability
    # in this case the relevant signal bus instance is named "bus".
    vars_of_interest = tz_fmu.find_vars(start_str='bus')

    # The investigated thermal zone model uses the signal bus as control interface.
    # It contains the in- and outputs.
    # bus.processVar:         zone temperature measurement
    # bus.controlOutput:      relative heating power
    # bus.disturbance[1]:     ambient air temperature

    # ############## Stepwise FMU Simulation #####################################

    # Initialize list for results
    # by reading the values of the relevant vars_of_interest from the fmu
    result_list = [tz_fmu.read_variables(vars_of_interest)]

    # simulation loop: Simulation the fmu stepwise for 12 hours and read results every step
    while tz_fmu.current_time < 12 * 3600:
        # perform simulation step
        tz_fmu.do_step()
        # read results and append to list
        res = tz_fmu.read_variables(vars_of_interest)
        result_list.append(res)
        print('Temperature: {}°C'.format(round(res['bus.processVar'] - 273.15)))

    # After 12 hours, the temperature reaches 18°C.
    # To turn on the heating the according variable is set
    tz_fmu.set_variables({'bus.controlOutput': 0.1})

    # The simulation is continued until the stop time is reached
    while not tz_fmu.finished:
        tz_fmu.do_step()
        result_list.append(tz_fmu.read_variables(vars_of_interest))

    # close fmu
    tz_fmu.close()

    # convert list of dicts to pandas datraframe
    sim_res_frame = pd.DataFrame(result_list)
    sim_res_frame.index = sim_res_frame['SimTime']

    # ########### Plotting ##########################################

    # Plotting the room temperature reveals
    # that turning on the heating could increase the temperature again
    x_values = sim_res_frame['SimTime'] / 3600
    fig, axes = plt.subplots(nrows=3, ncols=1)
    axes[0].set_title('Thermal zone with ideal heating')
    axes[0].plot(x_values, sim_res_frame['bus.processVar'] - 273.15, color='b')
    axes[0].hlines(18, x_values.iloc[0], x_values.iloc[-1], ls='--', color='r')
    axes[0].set_ylabel('T Zone / °C')
    axes[1].step(x_values, sim_res_frame['bus.controlOutput'], color='b')
    axes[1].set_ylabel('Rel. Power / -')
    axes[2].plot(x_values, sim_res_frame['bus.disturbance[1]'] - 273.15, color='b')
    axes[2].set_ylabel('T Ambient / °C')
    axes[2].set_xlabel('Time / h')

    for i in range(2):
        axes[i].set_xticklabels([])
    for ax in axes:
        ax.grid(True, 'both')
    plt.tight_layout()
    if with_plot:
        plt.show()


if __name__ == '__main__':
    main(
        with_plot=True,
        n_days=1
    )