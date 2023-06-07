import numpy as np
from typing import Dict, Tuple

from cryo_md.md_engine import run_MD_bias
from cryo_md.lklhood_and_grads import calc_gradient
from cryo_md.utils import check_config


def run_optimizer(
    init_models: np.ndarray,
    init_weights: np.ndarray,
    data: np.ndarray,
    prior_center: np.ndarray,
    config: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    check_config(config)

    trajectory = np.zeros(
        (config["opt_steps"] // config["stride"] + 1, *init_models.shape)
    )

    weight_traj = np.zeros(
        (config["opt_steps"] // config["stride"] + 1, *init_weights.shape)
    )

    trajectory[0] = init_models.copy()
    weight_traj[0] = init_weights.copy()

    curr_models = init_models.copy()
    curr_weights = init_weights.copy()

    counter = 1

    for i in range(config["opt_steps"]):
        samples = run_MD_bias(curr_models, prior_center, config)

        random_batch = np.arange(0, data.shape[0], 1)
        np.random.shuffle(random_batch)
        random_batch = random_batch[: config["batch_size"]]

        grad_strucs, grad_weights = calc_gradient(
            curr_models, samples, init_weights, data[random_batch], config
        )
        curr_models = curr_models + config["opt_step_size"] * grad_strucs
        curr_weights = curr_weights + config["opt_step_size"] * grad_weights
        curr_weights /= curr_weights.sum()

        if (i + 1) % config["stride"] == 0:
            trajectory[counter] = curr_models
            weight_traj[counter] = curr_weights
            counter += 1

    return (trajectory, weight_traj)
