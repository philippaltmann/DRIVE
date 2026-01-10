import src.environments.matrix_games as matrix_games
import src.environments.coin_game as coin_game
import src.environments.harvest as harvest
import numpy

def drift_function(function_name, constant=10, fraction=0.001):
    if function_name == "identity":
        return lambda t,x: x
    if function_name == "shift_pos":
        return lambda t,x: x + constant
    if function_name == "shift_neg":
        return lambda t,x: x - constant
    if function_name == "stepwise_increase":
        return lambda t,x: (numpy.floor(t/1000) + constant)*x
    if function_name == "scale_up":
        return lambda t,x: constant*x
    if function_name == "scale_down":
        return lambda t,x: (1.0/constant)*x
    if function_name == "linear":
        return lambda t,x: (1+t*fraction)*x
    if function_name == "quotient":
        return lambda t,x: (1.0/(t+1))*x
    if function_name == "exponential_increase":
        return lambda t,x: ((1+fraction)**t)*x
    if function_name == "exponential_decay":
        return lambda t,x: ((1+fraction)**(-t))*x
    if function_name == "cos_widened":
        double_frac = 2*fraction
        norm = 5000
        return lambda t,x: (fraction + t*numpy.cos(double_frac*t)*numpy.cos(double_frac*t)/norm)*x
    if function_name == "cos_damped":
        double_frac = 2*fraction
        norm = 5000
        return lambda t,x: (fraction + (norm-t)*numpy.cos(double_frac*t)*numpy.cos(double_frac*t)/norm)*x

def make(params):
    domain_name = params["domain_name"]
    if domain_name.startswith("Matrix-"):
        params["R_max"] = 3
        return matrix_games.make(params)
    if domain_name.startswith("CoinGame-"):
        params["R_max"] = 2
        return coin_game.make(params)
    if domain_name.startswith("Harvest-"):
        params["R_max"] = 0.25
        return harvest.make(params)
    raise ValueError("Unknown domain '{}'".format(domain_name))
