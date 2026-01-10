from settings import params
import src.domains as domains
import src.algorithms as algorithms
import src.experiments as experiments
import src.data as data
import sys

params["domain_name"] = sys.argv[1]
params["algorithm_name"] = sys.argv[2]
drift_function_name = sys.argv[3]
params["drift_function"] = domains.drift_function(drift_function_name)

env = domains.make(params)
env.reset()
controller = algorithms.make(params)

params["directory"] = params["output_folder"] + "/" + params["data_prefix_pattern"].\
    format(
        params["nr_agents"],\
        params["domain_name"],\
        drift_function_name,\
        params["algorithm_name"])
params["directory"] = data.mkdir_with_timestap(params["directory"])
experiments.run_training(env, controller, params)

