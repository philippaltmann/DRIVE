import src.controllers.controller as controller
import src.controllers.actor_critic as actor_critic
import src.controllers.mate as mate
import src.controllers.mediate as mediate
import src.controllers.drive as drive
import src.controllers.lio as lio
import src.controllers.ltos as ltos
import src.controllers.inequity_aversion as inequity_aversion

def make(params):
    algorithm_name = params["algorithm_name"]
    if algorithm_name == "Random":
        return controller.Controller(params)
    if algorithm_name == "IAC":
        return actor_critic.ActorCritic(params)
    if algorithm_name.startswith("LIO"):
        return lio.LIO(params)
    if algorithm_name.startswith("LToS"):
        return ltos.LToS(params)
    if algorithm_name.startswith("IA"):
        return inequity_aversion.InequityAversion(params)
    if algorithm_name.startswith("MATE"):
        params["mate_mode"] = "td_error"
        return mate.MATE(params)
    if algorithm_name == "MEDIATE": # Synchronous by default
        params["mate_mode"] = "td_error"
        params["consensus_on"] = True
        return mediate.MEDIATE(params)    
    if algorithm_name.startswith("DRIVE"):
        params["mate_mode"] = "td_error"
        if "-default" in algorithm_name: params["reciprocal_trust"] = False; params["response_mode"] = "mean"
        if "-instant" in algorithm_name: params["response_mode"] = "instant"
        if "-retrust" in algorithm_name: params["reciprocal_trust"] = True
        if "-ungated" in algorithm_name: params["mate_mode"] = "ungated"
        return drive.DRIVE(params)
    raise ValueError("Unknown algorithm '{}'".format(algorithm_name))