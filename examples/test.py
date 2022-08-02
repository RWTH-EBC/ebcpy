from ebcpy.simulationapi.fmu_handler import FMU

work_dir = pathlib.Path(__file__).parent.joinpath("results")
path = pathlib.Path(__file__).parent.joinpath("data", "ThermalZone_bus.fmu")
n_instances = 1

fmu_object = FMU(path, work_dir, n_instances)

