import src.environments.matrix_games as matrix_games
import src.environments.coin_game as coin_game
import src.environments.harvest as harvest
import numpy
 
def drift_function(params, modes=4, constant=10):
    """ modes: int(default 4): Number of changes 
    constant: int (default: 10): Constant shift """
    drift = params["drift"] if "drift" in params else "identity"
    n_epochs = params["nr_epochs"]
    eta = fraction = modes / n_epochs
    duration = n_epochs / modes 

    if drift == "identity": return lambda m,u: u
    if drift == "shift_pos": return lambda m,u: u + constant
    if drift == "shift_neg": return lambda m,u: u - constant
    if drift == "stepwise_increase":
        return lambda m,u: u * (numpy.floor(eta * m) + constant)
    if drift == "scale_up": return lambda m,u: u * constant
    if drift == "scale_down": return lambda m,u: u * (1.0 / constant)
    if drift == "linear": return lambda m,u: u * (eta * m + 1)
    if drift == "exponential_increase":
        return lambda m,u: u * numpy.exp(eta * m)
    if drift == "exponential_decay":
        return lambda m,u: u * numpy.exp(-eta * m)
    if drift == "cos_widened":        
        return lambda m,u: u * (m / n_epochs) * numpy.cos(2 * eta * m)**2
    if drift == "cos_damped":
        return lambda m,u: u * (1 - m / n_epochs) * numpy.cos(2 * eta * m)**2
    if drift == "noisy": # Per-agent per-step pertubtion
        sigma = params["d"]; print("sigma: ", sigma)       
        return lambda t,x: x + numpy.random.normal(0, sigma, size=(params['nr_agents'],))

def make_env(params):
    domain_name = params["domain_name"]
    if domain_name.startswith("Matrix-"):
        params["R_max"] = 3; params["d"] = 1
        return matrix_games.make(params)
    if domain_name.startswith("CoinGame-"):
        params["R_max"] = 2; params["d"] = 1 / int(domain_name[-1]) # 0.5
        return coin_game.make(params)
    if domain_name.startswith("Harvest-"):
        params["R_max"] = 0.25; params["d"] = 0.25; params['reciprocal_trust'] = True
        return harvest.make(params)
    raise ValueError("Unknown domain '{}'".format(domain_name))

def make(params):
    env = make_env(params); env.reset()
    params["drift_function"] = drift_function(params)
    return env, params
    