import src.controllers.controller as controller
import src.controllers.actor_critic as actor_critic
import src.controllers.mate as mate
import src.controllers.drive as drive
import src.controllers.lio as lio
import src.controllers.inequity_aversion as inequity_aversion

def make(params):
    algorithm_name = params["algorithm_name"]
    if algorithm_name == "Random":
        return controller.Controller(params)
    if algorithm_name == "IAC":
        return actor_critic.ActorCritic(params)
    if algorithm_name.startswith("LIO"):
        return lio.LIO(params)
    if algorithm_name.startswith("IA"):
        return inequity_aversion.InequityAversion(params)
    if algorithm_name.startswith("MATE-TD"):
        params["mate_mode"] = "td_error"
        return mate.MATE(params)
    if algorithm_name.startswith("DRIVE-TD"):
        params["mate_mode"] = "td_error"
        return drive.DRIVE(params)
    raise ValueError("Unknown algorithm '{}'".format(algorithm_name))