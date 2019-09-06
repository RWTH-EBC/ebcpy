# TODO Integrate based for Dymola and FMU in future.

"""Module to control multiprocessing of simulation of TEASER buildings."""
import os
import shutil
import numpy as np
import multiprocessing
from django import db
from dymola.dymola_interface import DymolaInterface


class WorkerSimulation(multiprocessing.Process):
    """Helper class to enable simulation in a JoinableQueue.

    This class inherits from multiprocessing.Proccess. Pass over your
    simulation function and  all parameters that are necessary for this
    function.

    Parameters
    ----------
    sim_function : simulate() function
    sim_part : Django QuerySet, NumPy Array or any other iterable
        Iterable collection of BuildingEnergy objects
    city_model : CityModel instance
        CityModel instance of the buildings that are simulated. Please ensure
        if you are using multiprocessing that all buildings are within the same
        city_model.
    model_path : str
        Path where Modelica model from TEASER is located. This is TEASER output
        path.
    results_path : str
        Path where Dymola results should be stored.
    start_time : int
        Start time of the simulation
    stop_time : int
        Stop time of the simulation
    output_interval : int
        Output interval of the simulation
    solver : string
        Used solver in Dymola. All Dymola solvers are supported(default:
        'Dassl'
    tolerance : float
        Tolerance of used solver
    process_number : int
        Counter of parallel processes, if multiprocessing is used. Default is
        None which indicates that no multuprocessing is used. Otherwise please
        set an individual number for every process that is used.
    result_queue : multiprocessing.JoinableQueue()
        JoinableQueue instance for each worker.

    """

    def __init__(
        self,
        sim_function,
        sim_part,
        city_model,
        model_path,
        results_path,
        start_time,
        stop_time,
        output_interval,
        method,
        tolerance,
        process_number,
        result_queue,
    ):
        """Init function of WorkerSimulation."""
        multiprocessing.Process.__init__(self)

        self.sim_function = sim_function
        self.sim_part = sim_part
        self.city_model = city_model
        self.model_path = model_path
        self.results_path = results_path
        self.start_time = start_time
        self.stop_time = stop_time
        self.output_interval = output_interval
        self.method = method
        self.tolerance = tolerance
        self.process_number = process_number
        self.result_queue = result_queue

    def run(self):
        """Run the specified simulation function with all parameters."""
        res_tmp = self.sim_function(
            sim_part=self.sim_part,
            city_model=self.city_model,
            model_path=self.model_path,
            results_path=self.results_path,
            start_time=self.start_time,
            stop_time=self.stop_time,
            output_interval=self.output_interval,
            method=self.method,
            tolerance=self.tolerance,
            process_number=self.process_number,
        )
        self.result_queue.put(res_tmp)


def simulate(
    sim_part,
    city_model,
    model_path,
    results_path,
    start_time,
    stop_time,
    output_interval,
    method,
    tolerance,
    process_number,
):
    """Simulate building models in serial using Dymola.

    This function simulates all buidlings given in the building_query_set in
    serial(one after another). This function can be used for parallel
    processing of simulation. There are no default values because it caused
    trouble with multiprocessing.

    Note: Please ensure that for all buildings corresponding models have been
    generated. Also ensure that all models are within the same
    city_model, otherwise this will neither work for single nor for
    multiprocessing.

    Parameters
    ----------
    sim_part : Django QuerySet, NumPy Array or any other iterable
        Iterable collection of BuildingEnergy objects
    city_model : CityModel instance
        CityModel instance of the buildings that are simulated. Please ensure
        if you are using multiprocessing that all buildings are within the same
        city_model.
    model_path : str
        Path where Modelica model from TEASER is located. This is TEASER output
        path.
    results_path : str
        Path where Dymola results should be stored.
    start_time : int
        Start time of the simulation
    stop_time : int
        Stop time of the simulation
    output_interval : int
        Output interval of the simulation
    solver : string
        Used solver in Dymola. All Dymola solvers are supported(default:
        'Dassl'
    tolerance : float
        Tolerance of used solver
    process_number : int
        Counter of parallel processes, if multiprocessing is used. Default is
        None which indicates that no multuprocessing is used. Otherwise please
        set an individual number for every process that is used.

    """
    SIMULATIONS_BEFORE_RESTART = 20
    db.connections.close_all()

    dir_result = os.path.join(results_path, city_model.name)
    dir_temp = os.path.join(dir_result, "temp" + str(process_number))
    if not os.path.exists(dir_result):
        os.makedirs(dir_result)
    if not os.path.exists(dir_temp):
        os.makedirs(dir_temp)

    for count, bldg in enumerate(sim_part):

        if count % SIMULATIONS_BEFORE_RESTART == 0:
            try:
                dymola.close()
            except NameError:
                pass
            dymola = DymolaInterface()
            dymola.openModel(
                path=os.path.join(os.environ["AIXLIB_LIBRARY_PATH"], "package.mo")
            )

            dymola.openModel(os.path.join(model_path, city_model.name, "package.mo"))

            dymola.cd(Dir=dir_temp)

        print("simulate building {} in progres".format(bldg.gmlid))
        model_name = "{}.{}.{}".format(city_model.name, bldg.gmlid, bldg.gmlid)
        dymola.translateModel(model_name)

        # Simulate the model

        output = dymola.simulateExtendedModel(
            problem=model_name,
            startTime=start_time,
            stopTime=stop_time,
            outputInterval=output_interval,
            method=method,
            tolerance=tolerance,
            resultFile=os.path.join(dir_result, bldg.gmlid),
        )

        if output[0] is False:
            print("Simulation failed. Below is the translation log.")
            log = dymola.getLastError()
            print(log)

    try:
        dymola.close()
    except NameError:
        pass
    shutil.rmtree(os.path.abspath(dir_temp))


def queue_simulation(
    sim_function,
    bldg_query_set,
    city_model,
    number_of_workers=multiprocessing.cpu_count() - 1,
    model_path=os.path.join(os.path.expanduser("~"), "TEASEROutput"),
    results_path=os.path.join(os.path.expanduser("~"), "djangoteaserout"),
    start_time=0.0,
    stop_time=31536000.0,
    output_interval=3600.0,
    method="Dassl",
    tolerance=0.0001,
):
    """Simulate multiple buildings in a JoinableQueue.

    This function simulates all buidlings given in the building_query_set. It
    uses JoinableQueue for multiprocessing.

    Note: Please ensure that for all buildings corresponding models have been
    generated. Also ensure that all models are within the same
    city_model, otherwise this will neither work for single nor for
    multiprocessing.

    Parameters
    ----------
    sim_function : simulate() function
    bldg_query_set : Django QuerySet, NumPy Array or any other iterable
        Iterable collection of BuildingEnergy objects
    city_model : CityModel instance
        CityModel instance of the buildings that are simulated. Please ensure
        if you are using multiprocessing that all buildings are within the same
        city_model.
    number_of_workers : int
        Number of workers used for the simulation. (default: number of
        physical processors - 1)
    model_path : str
        Path where Modelica model from TEASER is located. This is TEASER output
        path.
    results_path : str
        Path where Dymola results should be stored.
    start_time : int
        Start time of the simulation
    stop_time : int
        Stop time of the simulation
    output_interval : int
        Output interval of the simulation
    solver : string
        Used solver in Dymola. All Dymola solvers are supported(default:
        'Dassl'
    tolerance : float
        Tolerance of used solver

    """
    db.connections.close_all()
    workers = []

    simulation_parts = np.array_split(bldg_query_set, number_of_workers)

    result_queues = [multiprocessing.JoinableQueue()] * number_of_workers

    for i, (sim_part, result_queue) in enumerate(zip(simulation_parts, result_queues)):
        workers.append(
            WorkerSimulation(
                sim_function=sim_function,
                sim_part=sim_part,
                city_model=city_model,
                model_path=model_path,
                results_path=results_path,
                start_time=start_time,
                stop_time=stop_time,
                output_interval=output_interval,
                method=method,
                tolerance=tolerance,
                process_number=i,
                result_queue=result_queue,
            )
        )

    for w in workers:
        w.start()  # Start worker

    for w in workers:
        w.join()  # Block worker
