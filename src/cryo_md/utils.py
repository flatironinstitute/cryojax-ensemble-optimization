from typing import Dict


def check_config(config: Dict) -> None:
    req_keys = [
        "samp_steps",
        "samp_step_size",
        "md_force_constant",
        "samp_bias_force",
        "opt_steps",
        "opt_step_size",
        "batch_size",
        "sigma",
        "gamma",
        "delta_sigma",
        "stride",
    ]

    for key in req_keys:
        assert key in config.keys(), f"Error: missing key {key} in config"

    for key in config.keys():
        assert key in req_keys, f"Error: unknown key {key} in config"

    return


def help_config() -> None:
    help_message = (
        "samp_steps (int): number of steps to perform when sampling structures using MD. \n"
        + "samp_step_size (float): step size of the MD simulation. \n"
        + "md_force_constant (float): force constant for the 'MD forcefield'. \n"
        + "samp_bias_force (float): force constant of the biasing force (harmonic constraint) for the MD simulation. \n"
        + "opt_steps (int): number of optimization steps to perform (step = sampling + gradient descent). \n"
        + "opt_step_size (float): step size of the optimization step, equivalent to learning rate of the gradient. \n"
        + "batch_size (int): size of batch for stochastic gradient descent. \n"
        + "sigma (float): TO BE CHANGED - the standard deviation involved in the likelihood p(yi | sj). \n"
        + "gamma (float): TO BE CHANGED - gamma parameter in Eq 21. \n"
        + "delta_sigma (float): TO BE CHANGED - we model some delta functions as a gaussian, this is the standard deviation of those gaussians. \n"
        + "stride: how often you want a simulation step to be saved into the output."
    )

    print(help_message)

    return
