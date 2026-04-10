from pathlib import Path
from typing import Literal, overload

import yaml

from ._config_validators import (
    DatasetSimulatorConfig,
    EnsOptMDConfig,
    FlexibleFittingConfig,
    GMMFitConfig,
    ReweightingConfig,
)


@overload
def load_config(
    path_to_config: str | Path,
    config_mode: Literal["reweighting"],
) -> ReweightingConfig: ...


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
@overload
def load_config(
    path_to_config: str | Path,
    config_mode: Literal["flexible fitting"],
) -> FlexibleFittingConfig: ...


def load_config(
    path_to_config: str | Path,
    config_mode: Literal[
        "data simulation",
        "ensemble optimization",
        "gmm fitting",
        "flexible fitting",
        "reweighting",
    ],
) -> (
    DatasetSimulatorConfig
    | EnsOptMDConfig
    | GMMFitConfig
    | FlexibleFittingConfig
    | ReweightingConfig
):
    """
    Load a configuration file and parse it into the appropriate configuration object.

    **Arguments:**
    - `path_to_config`
        Path to the configuration file (YAML format).
    - `config_mode`
        Type of configuration to load. Must be one of "data simulation",
        "ensemble optimization", "gmm fitting", "flexible fitting", or "reweighting".

    **Returns:**
    - Config Object
        An instance of the appropriate configuration class based on the `config_mode`.
    """
    with open(path_to_config) as f:
        config_dict = yaml.safe_load(f)

    if config_mode == "data simulation":
        return DatasetSimulatorConfig(**config_dict)
    elif config_mode == "ensemble optimization":
        return EnsOptMDConfig(**config_dict)
    elif config_mode == "gmm fitting":
        return GMMFitConfig(**config_dict)
    elif config_mode == "flexible fitting":
        return FlexibleFittingConfig(**config_dict)
    elif config_mode == "reweighting":
        return ReweightingConfig(**config_dict)
    else:
        raise ValueError(
            f"Unknown config_mode: {config_mode}. Must be one of"
            + " 'data simulation', 'ensemble optimization',"
            " 'gmm fitting', 'flexible fitting', or 'reweighting'."
        )
