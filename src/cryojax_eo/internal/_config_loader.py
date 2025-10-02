from pathlib import Path
from typing_extensions import Literal

import yaml

from ._config_validators import DatasetGeneratorConfig, EnsOptMDConfig, GMMFitConfig


def load_config(
    path_to_config: str | Path,
    config_type: Literal["data generation", "ensemble optimization", "gmm fitting"],
) -> DatasetGeneratorConfig | EnsOptMDConfig | GMMFitConfig:
    """
    Load a configuration file and parse it into the appropriate configuration object.

    Parameters
    ----------
    path_to_config : str | Path
        Path to the configuration file (YAML format).
    config_type : Literal["data generation", "ensemble optimization", "gmm fitting"]
        Type of configuration to load. Must be one of "data generation",
        "ensemble optimization", or "gmm fitting".

    Returns
    -------
    DatasetGeneratorConfig | EnsOptMDConfig | GMMFitConfig
        Parsed configuration object.

    Raises
    ------
    ValueError
        If the provided config_type is not recognized.
    """
    with open(path_to_config, "r") as f:
        config_dict = yaml.safe_load(f)

    if config_type == "data generation":
        return DatasetGeneratorConfig(**config_dict)
    elif config_type == "ensemble optimization":
        return EnsOptMDConfig(**config_dict)
    elif config_type == "gmm fitting":
        return GMMFitConfig(**config_dict)
    else:
        raise ValueError(
            f"Unknown config_type: {config_type}. Must be one of"
            + " 'data generation', 'ensemble optimization', or 'gmm fitting'."
        )
