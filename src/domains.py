import src.environments.matrix_games as matrix_games
import src.environments.coin_game as coin_game
import src.environments.harvest as harvest
import numpy
 
def drift_function(function_name, params, modes=4, constant=10): # constant=constant, fraction=fraction
    """
    modes: int(default 4): Number of changes 
    constant: int (default: 10): Constant shift
    """
    n_epochs = params["nr_epochs"]
    eta = fraction = modes / n_epochs
    duration = n_epochs / modes 

    if function_name == "identity": return lambda m,u: u
    if function_name == "shift_pos": return lambda m,u: u + constant
    if function_name == "shift_neg": return lambda m,u: u - constant
    if function_name == "stepwise_increase":
        return lambda m,u: u * (numpy.floor(eta * m) + constant)
        return lambda m,u: u * ((numpy.floor(eta * m) + 1) * constant)
    if function_name == "scale_up": return lambda m,u: u * constant
    if function_name == "scale_down": return lambda m,u: u * (1.0 / constant)
    if function_name == "linear": return lambda m,u: u * (eta * m + 1)
    if function_name == "exponential_increase":
        return lambda m,u: u * numpy.exp(eta * m)
    if function_name == "exponential_decay":
        return lambda m,u: u * numpy.exp(-eta * m)
    if function_name == "cos_widened":        
        return lambda m,u: u * (m / n_epochs) * numpy.cos(2 * eta * m)**2
    if function_name == "cos_damped":
        return lambda m,u: u * (1 - m / n_epochs) * numpy.cos(2 * eta * m)**2
    if function_name == "noisy": # Per-agent per-step pertubtion
        sigma = params["d"]; print("sigma: ", sigma)       
        return lambda t,x: x + numpy.random.normal(0, sigma, size=(params['nr_agents'],))

def make(params):
    domain_name = params["domain_name"]
    if domain_name.startswith("Matrix-"):
        params["R_max"] = 3; params["d"] = 1
        return matrix_games.make(params)
    if domain_name.startswith("CoinGame-"):
        params["R_max"] = 2; params["d"] = 1 / int(domain_name[-1]) # 0.5
        return coin_game.make(params)
    if domain_name.startswith("Harvest-"):
        params["R_max"] = 0.25; params["d"] = 0.25
        return harvest.make(params)
    raise ValueError("Unknown domain '{}'".format(domain_name))
