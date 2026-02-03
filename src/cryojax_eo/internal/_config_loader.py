from pathlib import Path
from typing import Literal, overload

import yaml

from ._config_validators import DatasetSimulatorConfig, EnsOptMDConfig, GMMFitConfig


@overload
def load_config(
    path_to_config: str | Path,
    config_mode: Literal["data simulation"],
) -> DatasetSimulatorConfig: ...


@overload
def load_config(
    path_to_config: str | Path,
    config_mode: Literal["ensemble optimization"],
) -> EnsOptMDConfig: ...


@overload
def load_config(
    path_to_config: str | Path,
    config_mode: Literal["gmm fitting"],
) -> GMMFitConfig: ...


def load_config(
    path_to_config: str | Path,
    config_mode: Literal["data simulation", "ensemble optimization", "gmm fitting"],
) -> DatasetSimulatorConfig | EnsOptMDConfig | GMMFitConfig:
    """
    Load a configuration file and parse it into the appropriate configuration object.

    Parameters
    ----------
    path_to_config : str | Path
        Path to the configuration file (YAML format).
    config_mode : Literal["data simulation", "ensemble optimization", "gmm fitting"]
        Type of configuration to load. Must be one of "data simulation",
        "ensemble optimization", or "gmm fitting".

    Returns
    -------
    DatasetSimulatorConfig | EnsOptMDConfig | GMMFitConfig
        Parsed configuration object.

    Raises
    ------
    ValueError
        If the provided config_mode is not recognized.
    """
    with open(path_to_config) as f:
        config_dict = yaml.safe_load(f)

    if config_mode == "data simulation":
        return DatasetSimulatorConfig(**config_dict)
    elif config_mode == "ensemble optimization":
        return EnsOptMDConfig(**config_dict)
    elif config_mode == "gmm fitting":
        return GMMFitConfig(**config_dict)
    else:
        raise ValueError(
            f"Unknown config_mode: {config_mode}. Must be one of"
            + " 'data simulation', 'ensemble optimization', or 'gmm fitting'."
        )
