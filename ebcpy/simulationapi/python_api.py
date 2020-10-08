"""Module for classes using a fmu to
simulate models.."""

from ebcpy import simulationapi


class PYTHON_API(simulationapi.SimulationAPI):
    """
    Class for simulation using the fmpy library and
    a functional mockup interface as a model input.
    """

    # Default attributes
    sim_setup = {'startTime': 0.0,
                 'stopTime': 1.0,
                 'numberOfIntervals': 0,
                 'outputInterval': 1,
                 'solver': 'CVode',
                 'initialNames': [],
                 'initialValues': [],
                 'initialBoundaries': [],
                 "inputNames": [],
                 'resultNames': [],}

    # Dynamic setup of simulation setup
    number_values = [key for key, value in sim_setup.items() if
                     (isinstance(value, (int, float)) and not isinstance(value, bool))]

    def __init__(self, cd, model_name):
        """Instantiate class parameters"""
        super().__init__(cd, model_name)
        if not model_name.lower().endswith(".?"): #to-do: add filetype
            raise ValueError("{} is not a valid ANN file!".format(model_name))

        # Extract Inputs, Outputs & Tuner (lists from parent classes will be appended)
        # possible to add lists with inputs in json. Therefore extend the json file

        # self.model_inp =
        # self.model_out =
        # self.model_tuner_names =
        # self.model_tuner_bounds =         # add tuner bounds as tuple: [(min, max), (min, max),...]
        # self.model_tuner_initialvalues =

    def set_cd(self, cd):
        """
        Set current working directory for storing files etc.
        :param str,os.path.normpath cd:
            New working directory
        :return:
        """
        os.makedirs(cd, exist_ok=True)
        self.cd = cd

    def simulate(self, meas_input_data, **kwargs):              # %%%% TO-DO: Automatisieren. Anpassen auf InfluxDB.
        """
        Simulate current simulation-setup.

        :param ?:

        :return dataframe sim_target_data:
            Pandas.Dataframe of simulated target values
        """

        # %%%% TO-DO: Add simulation of Python model here %%%%%

        # Dict with all tunerparameternames & -values
        start_values = {self.sim_setup["initialNames"][i]: value
                        for i, value in enumerate(self.sim_setup["initialValues"])}

        return df

    def do_step(self):
        # check if stop time is reached
        if self.current_time < self.stop_time:
            pass
            #..to add...

    def close(self):
        """
        Closes the fmu.
        :return:
            True on success
        """
        pass
        #print("What to close??")

    def test(self):
        pass