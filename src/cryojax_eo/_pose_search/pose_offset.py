import jax.numpy as jnp
from jaxtyping import Array, Float


def compute_correlation_at_optimal_offset(
    image_shifted: Float[Array, "H W"],
    image_ref: Float[Array, "H W"],
    coordinate_grid_in_angstroms: Float[Array, "H W 2"],
    shift_search_area: Float[Array, "H W"] | None = None,
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
    - `shift_search_area`:
        Optional mask, on the same grid as `coordinate_grid_in_angstroms`,
        restricting the region of shifts that is searched. Shifts where the
        mask is (numerically) zero are excluded. If `None`, all shifts are
        searched.
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
    # The zero-lag component comes out of the FFT in the corner, so center it
    # to match the (fftshifted) coordinate grid and the shift search area
    abs_cross_corr = jnp.fft.fftshift(abs_cross_corr)

    if shift_search_area is not None:
        abs_cross_corr = jnp.where(shift_search_area > 1e-3, abs_cross_corr, -jnp.inf)

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
    # image_shifted_fft = cxim.rfftn(image_shifted)
    # image_ref_fft = cxim.rfftn(image_ref)
    image_shifted_fft = jnp.fft.rfftn(image_shifted)
    image_ref_fft = jnp.fft.rfftn(image_ref)
    cross_corr_fft = image_shifted_fft * image_ref_fft.conj()
    return jnp.fft.irfftn(cross_corr_fft)
