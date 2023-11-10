from nuplan.planning.script.builders.planner_builder import build_planners
from omegaconf import DictConfig, OmegaConf
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
import time #for perfroamcne timing it seems
import pytorch_lightning as pl
from nuplan.planning.script.builders.utils.utils_config import update_config_for_simulation
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.script.builders.worker_pool_builder import build_worker
from nuplan.planning.script.builders.folder_builder import build_simulation_experiment_folder
from nuplan.planning.script.builders.simulation_callback_builder import build_simulation_callbacks
from nuplan.planning.training.callbacks.profile_callback import ProfileCallback
import pathlib
from nuplan.planning.script.builders.simulation_builder import build_simulations
from nuplan.planning.script.builders.scenario_building_builder import build_scenario_builder
from nuplan.planning.simulation.runner import run_planner_through_scenarios
import logging

##################### From the PREPARE THE SIMULATION CONFIG cell #####################
# Location of path with all simulation configs
CONFIG_PATH = '../nuplan/planning/script/config/simulation'
CONFIG_NAME = 'default_simulation'

# Select the planner and simulation challenge
PLANNER = 'simple_planner'  # [simple_planner, ml_planner]
CHALLENGE = 'challenge_4_closed_loop_reactive_agents'  # [challenge_1_open_loop_boxes, challenge_3_closed_loop_nonreactive_agents, challenge_4_closed_loop_reactive_agents]
DATASET_PARAMS = [
    'scenario_builder=nuplan_mini',  # use nuplan mini database
    'scenario_builder/nuplan/scenario_filter=all_scenarios',  # initially select all scenarios in the database
    'scenario_builder.nuplan.scenario_filter.scenario_types=[ego_at_pudo]',#[nearby_dense_vehicle_traffic, ego_at_pudo, ego_starts_unprotected_cross_turn, ego_high_curvature]',  # select scenario types
    'scenario_builder.nuplan.scenario_filter.limit_scenarios_per_type=1',  # default 10 scenarios per scenario type, Mike adjusted down to 1 for simplicity, feel free to increase
    'scenario_builder.nuplan.scenario_filter.subsample_ratio=0.05',  # subsample 20s scenario from 20Hz to 1Hz
]

# Name of the experiment
EXPERIMENT = 'simulation_simple_experiment_mike_ML_C4'#custom experiment name

# Initialize configuration management system
hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
hydra.initialize(config_path=CONFIG_PATH)

# Compose the configuration
cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[
    f'experiment_name={EXPERIMENT}',
    f'group={SAVE_DIR}',
    f'planner={PLANNER}',
    f'+simulation={CHALLENGE}',
    *DATASET_PARAMS,
])
logger = logging.getLogger(__name__)

#########################################################################################################


##################### From run_simulation.py ###############################################################

from typing import List, Union
# DEFINE THE run_simulation function (really just pulled from the file)
def run_simulation(cfg: DictConfig, planner: Union[AbstractPlanner, List[AbstractPlanner]]) -> None:
    """
    Execute all available challenges simultaneously on the same scenario. Helper function for main to allow planner to
    be specified via config or directly passed as argument.
    :param cfg: Configuration that is used to run the experiment.
        Already contains the changes merged from the experiment's config to default config.
    :param planner: Pre-built planner(s) to run in simulation. Can either be a single planner or list of planners
    """

    # Make sure a planner is specified and that two separate planners are not being specified from both arg and config.
    if 'planner' in cfg.keys():
        raise ValueError("Planner specified via both config and argument. Please only specify one planner.")
    if planner is None:
        raise TypeError("Planner argument is None.")

    start_time = time.perf_counter()

    # Fix random seed
    pl.seed_everything(cfg.seed, workers=True)

    # Update and override configs for simulation
    update_config_for_simulation(cfg=cfg)

    # Configure logger
    build_logger(cfg)

    # Construct builder
    worker = build_worker(cfg)

    # Create output storage folder
    build_simulation_experiment_folder(cfg=cfg)

    # Simulation Callbacks
    output_dir = pathlib.Path(cfg.output_dir)
    callbacks = build_simulation_callbacks(cfg=cfg, output_dir=output_dir)

    # Create profiler if enabled
    profiler = None
    if cfg.enable_profiling:
        logger.info("Profiler is enabled!")
        profiler = ProfileCallback(output_dir=output_dir)

    if profiler:
        # Profile the simulation construction
        profiler.start_profiler("building_simulation")

    # Build scenario builder
    scenario_builder = build_scenario_builder(cfg=cfg)
    # Construct simulations
    if isinstance(planner, AbstractPlanner):
        planner = [planner]
    simulations = build_simulations(cfg=cfg, callbacks=callbacks, scenario_builder=scenario_builder, worker=worker,
                                    planners=planner) #RETURNS A LIST of simulation objects
    assert len(simulations) > 0, 'No scenarios found to simulate!'

    #Mike play: let's see what we can access from one of these simulations:
    scen0 = simulations[0].scenario
    print(f"Check: #of scenario iterations in simulations[0].scenario:  {scen0.get_number_of_iterations()}")
    ## Breakdown of structures
    #run_simulation - fed with whole config and the planner(s) you want to use (from earlier setup steps)
    #    builds a worker (multithreading and gpu stuff)
    #    builds simulations (each has one scenario)
    #        each scenario has 0+ *iterations* which are the steps of the scenario
    #        the simulation.py's *step() function simulates one of these *iterations*
            
    
    if profiler:
        # Stop simulation construction profiling
        profiler.save_profiler("building_simulation")
        # Start simulation running profiling
        profiler.start_profiler("running_simulation")

    logger.info("Running simulation...")
    #make an output file:
    f = open("diaz_trajectories_simOuts.csv", "w")
    #print header:
    f.write(f'timestamp,prediction_timestamp,x_center,y_center\n')
    f.close()
    #MIKE DIAZ:  In the run_planner_through_scenarios function you'll dig till simulation.py's *step()* function where i've made changes to access the states
    run_planner_through_scenarios(simulations=simulations,
                                  worker=worker,
                                  num_gpus=cfg.number_of_gpus_used_for_one_simulation,
                                  num_cpus=cfg.number_of_cpus_used_for_one_simulation,
                                  exit_on_failure=cfg.exit_on_failure)
    logger.info("Finished running simulation!")

    # Save profiler
    if profiler:
        profiler.save_profiler("running_simulation")

    end_time = time.perf_counter()
    elapsed_time_s = end_time - start_time
    time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time_s))
    logger.info(f"Simulation duration: {time_str} [HH:MM:SS]")
    
#####################end from run_simulation.py ###############################################################


##################### from run_simulation.py, more ###############################################################
    
#print(cfg)    
# Build planners.
if 'planner' not in cfg.keys():
    raise KeyError("Planner not specified in config. Please specify a planner using 'planner' field.")
print("Building planners...")
planners = build_planners(cfg.planner)
print("Building planners...DONE!")

# Remove planner from config to make sure run_simulation does not receive multiple planner specifications.
OmegaConf.set_struct(cfg, False)
cfg.pop('planner')
OmegaConf.set_struct(cfg, True)

# Execute simulation with preconfigured planner(s).
run_simulation(cfg=cfg, planner=planners)




