def validate_generic_config_req(config: dict, reference: dict) -> None:
    """
    Validate a config dictionary against a reference dictionary.

    Parameters
    ----------
    config : dict
        The dictionary to validate.
    reference : dict
        The reference dictionary to validate against.

    Raises
    ------
    ValueError
        If a key in reference is not present in config.
    ValueError
        If the type of a key in config does not match the type of the corresponding key in reference.

    Returns
    -------
    None
    """  # noqa: E501
    for key in reference:
        if key not in config:
            raise ValueError(f"Missing key in config: {key}")
        if not isinstance(config[key], reference[key]):
            raise ValueError(
                f"Invalid type for key {key} in config: {type(config[key])}"
            )
    return


def validate_generic_config_opt(config: dict, reference: dict) -> dict:
    """
    Validate a config dictionary with optional parameters against a reference dictionary.

    Parameters
    ----------
    config : dict
        The dictionary to validate.
    reference : dict
        The reference dictionary to validate against.

    Raises
    ------
    ValueError
        If the type of a key in config does not match the type of the corresponding key in reference.

    Returns
    -------
    None
    """  # noqa: E501
    for key in reference:
        if key not in config:
            config[key] = reference[key][1]
        elif not isinstance(config[key], reference[key][0]):
            raise ValueError("{} must be of type {}".format(key, reference[key][0]))

    return config
