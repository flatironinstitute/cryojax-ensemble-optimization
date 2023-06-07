import numpy as np
from typing import Dict


def run_MD_bias(
    init_models: np.ndarray, prior_center: np.ndarray, config: Dict
) -> np.ndarray:
    samples = np.zeros((config["samp_steps"] + 1, *init_models.shape))
    samples[0] = init_models.copy()

    for i in range(config["samp_steps"]):
        samples[i + 1] = (
            samples[i]
            + np.sqrt(2 * config["samp_step_size"])
            * np.random.randn(*init_models.shape)
            - config["samp_step_size"]
            * config["samp_bias_force"]
            * (samples[i] - samples[0])
            - config["samp_step_size"]
            * config["md_force_constant"]
            * (samples[i] - prior_center[None, :])
        )

    return samples
