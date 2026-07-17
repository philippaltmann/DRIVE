import argparse
import random

import numpy as np
import torch as th

from settings import params
import src.domains as domains
import src.algorithms as algorithms
import src.experiments as experiments
import src.data as data

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an algorithm on a selected domain and reward drift.")
    parser.add_argument("algorithm_name", type=str, help="Name of the learning algorithm.")
    parser.add_argument("domain_name", type=str, help="Name of the training domain.")
    parser.add_argument("-d", "--drift", type=str, default="identity", help="Name of the reward drift function.")
    parser.add_argument("-s", "--seed", type=int, default=None, help="Random seed. By default, no explicit seed is set.")
    parser.add_argument("-e", "--nr-epochs", type=int, default=None, help=("Number of training epochs. By default, uses 4000 epochs."))
    parser.add_argument("-o", "--output-folder", type=str, default="output", help="Folder in which results are stored.")
    return parser.parse_args()

def seed_everything(seed: int | None) -> None:
    if seed is None: return
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    print(f"Seeded using {seed}")

if __name__ == "__main__":
    args = parse_args()
    params["domain_name"] = args.domain_name
    params["algorithm_name"] = args.algorithm_name
    params["drift"] = args.drift
    params["output_folder"] = args.output_folder
    params["nr_epochs"] = args.nr_epochs or params["nr_epochs"]
    seed_everything(args.seed)

    print(f"Running {params['domain_name']} with {params['algorithm_name']} for {params['nr_epochs']} epochs.")

    env, params = domains.make(params)
    controller = algorithms.make(params)
    params["directory"] = data.mkdir(params, args.seed)

    experiments.run_training(env, controller, params)
