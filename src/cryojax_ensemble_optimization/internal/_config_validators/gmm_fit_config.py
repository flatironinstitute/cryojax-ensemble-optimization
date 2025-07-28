from pydantic import (
    BaseModel,
    Field,
    PositiveFloat,
    PositiveInt,
)


class GMMFitConfig(BaseModel, extra="forbid"):
    box_size: PositiveInt = Field(
        default=128,
        description="Size of the box in pixels."
        + "This is used to compute the density used to fit the GMM",
    )
    voxel_size: PositiveFloat = Field(
        default=2.0,
        description="Size of the voxel in Angstroms."
        + "This is used to compute the density used to fit the GMM",
    )
    fit_selection_string: str = Field(
        default='name "C2"',
        description="Selection string to select the atoms used to define the GMM."
        + "This is used to select the atoms from the reference file.",
    )
    init_log_amp: float = Field(
        default=40.0,
        description="Initial log amplitude for the GMM fitting."
        + "This is used to initialize the GMM fitting.",
    )
    init_log_var: float = Field(
        default=1.0,
        description="Initial log variance for the GMM fitting."
        + "This is used to initialize the GMM fitting.",
    )
    n_gaussians_per_bead: PositiveInt = Field(
        default=1,
        description="Number of Gaussians per bead in the GMM fitting."
        + "This is used to define the number of Gaussians per bead in the GMM fitting.",
    )
    atol: float = Field(
        default=1e-3,
        description="Absolute tolerance for the GaussNewton Algorithm used to fit the GMM."  # noqa
        + "This is used to define the convergence criteria for the GMM fitting.",
    )
    rtol: float = Field(
        default=1e-3,
        description="Relative tolerance for the GaussNewton Algorithm used to fit the GMM."  # noqa
        + "This is used to define the convergence criteria for the GMM fitting.",
    )
    max_steps: PositiveInt = Field(
        default=500,
        description="Maximum number of steps for the GaussNewton Algorithm used to fit the GMM."  # noqa
        + "This is used to define the maximum number of steps for the GMM fitting.",
    )
