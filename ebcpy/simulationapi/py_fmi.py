"""Module for classes using a fmu to
simulate models."""

from ebcpy import simulationapi
import fmpy
import shutil
import os


class FMU_API(simulationapi.SimulationAPI):
    """
    Class for simulation using the fmpy library and
    a functional mockup interface as a model input.
    """

    sim_setup = {'startTime': 0.0,
                 'stopTime': 1.0,
                 'numberOfIntervals': 0,
                 'outputInterval': 1,
                 'solver': 'CVode',
                 'initialNames': [],
                 'initialValues': [],
                 'finalNames': []}

    def __init__(self, cd, model_name):
        """Instantiate class parameters"""
        super().__init__(cd, model_name)
        if not model_name.lower().endswith(".fmu"):
            raise ValueError("{} is not a valid fmu file!".format(model_name))

    def close(self):
        """
        Closes the fmu.
        :return:
            True on success
        """
        raise NotImplementedError

    def set_cd(self, cd):
        """
        Set current working directory for storing files etc.
        :param str,os.path.normpath cd:
            New working directory
        :return:
        """
        self.cd = cd

    def simulate(self, fail_on_error=True):
        """
        Simulate current simulation-setup.

        :param str,os.path.normpath savepath_files:
            Savepath were to store result files of the simulation.
        :return:
            Filepath of the mat-file.
        """
        start_values = {self.sim_setup["initialNames"][i]: value
                        for i, value in enumerate(self.sim_setup["initialValues"])}
        try:
            res = fmpy.simulate_fmu(
                     self.model_name,
                     validate=True,
                     start_time=self.sim_setup["startTime"],
                     stop_time=self.sim_setup["stopTime"],
                     solver=self.sim_setup["solver"],
                     step_size=self.sim_setup["numberOfIntervals"],
                     relative_tolerance=None,
                     output_interval=self.sim_setup["outputInterval"],
                     record_events=False,
                     fmi_type=None,
                     start_values=start_values,
                     apply_default_start_values=False,
                     input=None,
                     output=self.sim_setup["finalNames"],
                     timeout=None,
                     debug_logging=False,
                     visible=False,
                     logger=None,
                     fmi_call_logger=None,
                     step_finished=None,
                     model_description=None,
                     fmu_instance=None)
        except Exception as error:
            res = None
            print(f"[SIMULATION ERROR] Error occured while running FMU: \n {error}")
            if fail_on_error:
                raise error

        return res

    def set_sim_setup(self, sim_setup):
        """
        Alter the simulation setup by changing the setup-dict.

        :param sim_setup:
        """
        _diff = set(sim_setup.keys()).difference(self.sim_setup.keys())
        if _diff:
            raise KeyError("The given sim_setup contains the following keys ({}) which are "
                           "not part of the fmu sim_setup.".format(" ,".join(list(_diff))))
        _number_values = ["startTime", "stopTime", "numberOfIntervals",
                          "outputInterval"]
        for key, value in sim_setup.items():
            if key in _number_values:
                _ref = (float, int)
            else:
                _ref = type(self.sim_setup[key])
            if isinstance(value, _ref):
                self.sim_setup[key] = value
            else:
                raise TypeError("{} is of type {} but should be"
                                " type {}".format(key, type(value).__name__, _ref))


if __name__=="__main__":
    path = r"D:\pme-fwu\00_testzone\test_model.fmu"
    p = FMU_API(cd=os.path.dirname(path), model_name=path)
    INP_PARAM_NAMES = ["optimizationVariables.V_PS",
                       "optimizationVariables.V_TWWS",
                       "optimizationVariables.Q_HR_Nom",
                       "optimizationVariables.Q_HP_Nom"]
    p.set_sim_setup({"stopTime": 1000000,
                     "finalNames": ["W_el"],
                     "initialNames": INP_PARAM_NAMES,
                     "initialValues": [3000, 100, 100, 3000],
                     "outputInterval": 3600})
    res = p.simulate()
    print(res.shape)
    import matplotlib.pyplot as plt
    plt.plot([e[1] for e in res])
    plt.show()