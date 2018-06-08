from .database import DatabaseManager
from .runner import SimulationRunner
from .parallelrunner import ParallelRunner
from git import Repo
from copy import deepcopy
from tqdm import tqdm
from random import shuffle
import numpy as np
import xarray as xr


class CampaignManager(object):
    """
    The main Simulation Execution Manager class can be used to load, save,
    execute and access the results of simulation campaigns.
    """

    #######################################
    # Campaign initialization and loading #
    #######################################

    def __init__(self, campaign_db, campaign_runner):
        """
        Initialize the Simulation Execution Manager.
        """
        self.db = campaign_db
        self.runner = campaign_runner

    @classmethod
    def new(cls, path, script, campaign_dir, runner='SimulationRunner'):
        """
        Initialize a campaign database based on a script and ns-3 path.

        runner can be either SimulationRunner (default) or ParallelRunner
        """
        # Create a runner for the desired configuration
        if runner == 'SimulationRunner':
            runner = SimulationRunner(path, script)
        elif runner == 'ParallelRunner':
            runner = ParallelRunner(path, script)
        else:
            raise ValueError('Unknown runner')

        # Get list of available parameters
        params = runner.get_available_parameters()

        # Repository check
        # TODO Make sure there are no staged/unstaged changes
        # Get current commit
        commit = Repo(path).head.commit.hexsha

        # Create a database manager from configuration
        config = {
            'script': script,
            'path': path,
            'params': params,
            'commit': commit,
            'campaign_dir': campaign_dir
        }

        db = DatabaseManager.new(config, campaign_dir)

        return cls(db, runner)

    @classmethod
    def load(cls, filename, runner='SimulationRunner'):
        """
        Read a filename and load the corresponding campaign database.
        """
        # Read from database
        db = DatabaseManager.load(filename)

        # Create a runner for the desired configuration
        if runner == 'SimulationRunner':
            runner = SimulationRunner(db.get_path(), db.get_script())
        elif runner == 'ParallelRunner':
            runner = ParallelRunner(db.get_path(), db.get_script())
        else:
            raise ValueError('Unknown runner')

        return cls(db, runner)

    ######################
    # Simulation running #
    ######################

    def run_simulations(self, param_list, verbose=True):
        """
        Run several simulations specified by a list of parameters.

        This function does not verify whether we already have the required
        simulations in the database - it just runs all the parameter
        combinations that are specified in the list.
        """
        # Compute next RngRun value
        next_run = self.db.get_next_rngrun()
        for idx, param in enumerate(param_list):
            param['RngRun'] = next_run + idx

        # Shuffle simulations
        # This mixes up long and short simulations, and gives better time
        # estimates
        shuffle(param_list)

        # Offload simulation execution to self.runner
        # Note that this only creates a generator for the results, no
        # computation is performed on this line
        results = self.runner.run_simulations(param_list,
                                              self.db.get_data_dir())

        for result in tqdm(results, total=len(param_list), unit='simulation',
                           desc='Running simulations'):
            # Insert result object in db
            self.db.insert_result(result)

    def get_missing_simulations(self, param_list, runs):
        """
        Return a list of the simulations among the required ones that are not
        available in the database.

        Args:
            param_list (list): A list of dictionaries containing all the
                parameters combinations.
            runs (int): An integer representing how many repetitions are wanted
                for each parameter combination.
        """
        params_to_simulate = []

        for param_comb in param_list:
            available_sims = self.db.get_results(param_comb)
            needed_runs = runs - len(available_sims)
            # Here it's important that we make copies of the dictionaries, so
            # that if we modify one we don't modify the others. This is
            # necessary because after this step, typically, we will add the
            # RngRun key which must be different for each copy.
            params_to_simulate += [deepcopy(param_comb) for i in
                                   range(needed_runs)]

        return params_to_simulate

    def run_missing_simulations(self, param_list, runs):
        """
        Run the simulations from the parameter list that are not yet available
        in the database.

        This function makes sure that we have at least runs replications for
        each parameter combination.
        """
        self.run_simulations(self.get_missing_simulations(param_list, runs))

    #####################
    # Result management #
    #####################

    def get_results_as_numpy_array(self, parameter_space,
                                   stdout_parsing_function,
                                   run_averaging_function=None):
        """
        Return the results relative to the desired parameter space in the form
        of a numpy array.
        """
        return np.squeeze(np.array(self.get_space({}, parameter_space,
                                                  stdout_parsing_function,
                                                  run_averaging_function)))

    def get_results_as_xarray(self, parameter_space,
                              stdout_parsing_function,
                              run_averaging_function=None):
        """
        Return the results relative to the desired parameter space in the form
        of an xarray data structure.
        """
        np_array = np.squeeze(np.array(self.get_space({}, parameter_space,
                                                      stdout_parsing_function,
                                                      run_averaging_function)))

        # Create a parameter space only containing the variable parameters
        clean_parameter_space = {}
        for key, value in parameter_space.items():
            if isinstance(value, list) and len(value) > 1:
                clean_parameter_space[key] = value

        clean_parameter_space['runs'] = range(np_array.shape[-1])

        xr_array = xr.DataArray(np_array, coords=clean_parameter_space,
                                dims=list(clean_parameter_space.keys()))

        return xr_array

    def get_space(self, current_query, param_space, stdout_parsing_function,
                  run_averaging_function):
        # print("Parameter space: %s" % param_space)
        # print("Current query: %s" % current_query)
        if not param_space:
            # print("Querying database with query:\n%s" % current_query)
            results = self.db.get_results(current_query)
            parsed = []
            for r in results:
                parsed.append(stdout_parsing_function(r['stdout']))

            # print("Runs: %s" % parsed)
            if run_averaging_function is not None:
                return run_averaging_function(parsed)
            else:
                return parsed

        space = []
        [key, value] = list(param_space.items())[0]
        for v in value:
            # print("Key: %s, Value: %s" % (key, v))
            next_query = deepcopy(current_query)
            next_query[key] = v
            next_param_space = deepcopy(param_space)
            del(next_param_space[key])
            space.append(self.get_space(next_query, next_param_space, stdout_parsing_function, run_averaging_function))
        return space

    #############
    # Utilities #
    #############

    def __str__(self):
        return "--- Campaign info ---\n%s\n------------" % self.db
