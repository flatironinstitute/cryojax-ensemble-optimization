import cryojax.ndimage as cxim
import jax.numpy as jnp
from jaxtyping import Array, Float


def compute_correlation_at_optimal_offset(
    image_shifted: Float[Array, "H W"],
    image_ref: Float[Array, "H W"],
    coordinate_grid_in_angstroms: Float[Array, "H W 2"],
) -> tuple[Float[Array, ""], Float[Array, "2"]]:
    """
    Estimates the correlation between two images at the optimal shift
    using Plancherel's theorem and Fourier's convolution theorem.

    **Arguments:**
    - `image_shifted`:
        Image that has been shifted.
    - `image_ref`:
        Reference centered image.
    - `coordinate_grid_in_angstroms`:
        Coordinate grid that maps pixel indices to positions in angstroms.
    **Returns:**
    - `loss`:
        Negative correlation at the optimal shift.
    - `shift`:
        Shift that best aligns ``image_shifted`` to
        ``image_ref`` in the phase-correlation sense.
    """
    abs_cross_corr = jnp.abs(
        _compute_cross_correlation_image(
            image_shifted / jnp.linalg.norm(image_shifted),
            image_ref / jnp.linalg.norm(image_ref),
        )
    )

    # Peak gives the shift. can fit for subpixel accuracy if needed
    max_idx = jnp.unravel_index(jnp.argmax(abs_cross_corr), abs_cross_corr.shape)
    optimal_shift = coordinate_grid_in_angstroms[max_idx]

    return (
        jnp.max(abs_cross_corr),
        optimal_shift,
    )


def _compute_cross_correlation_image(image_shifted, image_ref):
    """Compute cross-correlation image between two input images.
    Using Fourier's convolution theorem.
    """
    image_shifted_fft = cxim.rfftn(image_shifted)
    image_ref_fft = cxim.rfftn(image_ref)
    cross_corr_fft = image_shifted_fft * image_ref_fft.conj()
    return cxim.irfftn(cross_corr_fft)
