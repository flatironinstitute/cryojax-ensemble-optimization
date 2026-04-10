"""
This was implemented by Luke Evans, and was ported from repo:
https://github.com/aevans1/mult_grad_population_calibration/tree/main

This method has been used in the following publications:

https://www.biorxiv.org/content/10.1101/2025.03.27.644168v1
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


class _MultGradWeightCarry(eqx.Module):
    weights: Float[Array, " n_nodes"]
    gap: Float[Array, ""]
    n_iters: Int[Array, ""]
    # these two are constants
    likelihood: Float[Array, "n_data n_nodes"]
    gap_scale: Float[Array, ""]
    tol: Float[Array, ""]


@eqx.filter_jit
def multiplicative_gradient(
    log_likelihood_matrix: Float[Array, "n_data n_nodes"],
    tol: Float[Array, ""],
    max_iter: Int[Array, ""],
) -> tuple[Float[Array, " n_nodes"], Int[Array, ""], Float[Array, ""]]:
    """
    optimizes the weights with the multiplicative gradient method.

    ** Arguments:**

    - `log_likelihood_matrix`:
        Likelihood matrix computed from `n_data` data points and `n_nodes` nodes.
        Where nodes are representations of atomic structures (atomic models or volumes).
    - `tol`:
        Tolerance for the stopping criteria
    - `max_iter`:
        Maximum iterations if stopping criteria isn't met

    ** Returns:**

    - `weights:
        optimized weights for the nodes, based on the multiplicative gradient method.
    - `n_iters`:
        number of iterations taken to converge
    - `final_gap`:
        final gap value, which is a proxy for convergence.
    """

    def body_fn(carry: _MultGradWeightCarry):
        grad = _compute_grad(carry.weights, carry.likelihood)
        gap = _scaled_gap(grad, carry.weights, carry.gap_scale)
        weights = _update_weights(carry.weights, grad)
        return _MultGradWeightCarry(
            weights=weights,
            gap=gap,
            n_iters=carry.n_iters + 1,
            likelihood=carry.likelihood,
            gap_scale=carry.gap_scale,
            tol=carry.tol,
        )

    def cond_fn(carry: _MultGradWeightCarry):
        cond1 = jnp.greater_equal(carry.gap, carry.tol)
        cond2 = jnp.less(carry.n_iters, max_iter)
        return jnp.logical_and(cond1, cond2)

    n_nodes = log_likelihood_matrix.shape[1]

    # Initialize weights
    weights = (1 / n_nodes) * jnp.ones(n_nodes)

    # Convert log likelihood to likelihood via "soft-max"-ish operation
    likelihood = _normalize_log_likelihood_to_likelihood(log_likelihood_matrix)

    # Initialize scaling for gap stopping criteria
    gap_scale = _scaled_gap(_compute_grad(weights, likelihood), weights, scale=1.0)

    carry = _MultGradWeightCarry(
        weights=weights,
        gap=gap_scale,
        n_iters=jnp.asarray(0),
        likelihood=likelihood,
        gap_scale=gap_scale,
        tol=tol,
    )

    carry = jax.lax.while_loop(cond_fn, body_fn, carry)

    return carry.weights, carry.n_iters, carry.gap


def _normalize_log_likelihood_to_likelihood(
    log_likelihood: Float[Array, "n_data n_nodes"],
) -> Float[Array, "n_data n_nodes"]:
    """
    Subtracts the largest entry from each row of  the log likelihood.
    This is for stability, before transforming to likelihood (like in a soft-max)
    The gradient is invariant to row scaling of likelihood, so this is valid.
    With this normalizing, we avoid working in log space for the grad and loss.

    ** Arguments:**
    `log_likelihood`:
        log-likelihood matrix computed from `n_data` data points and `n_nodes` nodes.
        Where nodes are representations of atomic structures (atomic models or volumes).

    ** Returns:**
    `likelihood`:
        Normalized likelihood matrix, where each row has had the max entry subtracted,
        and then exponentiated.
    """
    log_likelihood -= jnp.amax(log_likelihood, axis=1)[:, None]
    likelihood = jnp.exp(log_likelihood)
    return likelihood


def _compute_grad(
    weights: Float[Array, " n_nodes"], likelihood: Float[Array, "n_data n_nodes"]
) -> Float[Array, " n_nodes"]:
    """
    Evaluate the gradient of the log-likelihood of the data given the weights.

    This computes the "probabilistic model" for the data prob density with weights w
    - sum_j p(y_i |x_j) w_j
    And then computes the gradient of (1/n_data)*sum_i log (sum_j p(y_i|x_j) w_j):
    - (1/n_data)*sum_i ((p(y_i|x_j) / sum_k p(y_i | x_k) w_j))

    Parameters
    ----------
    weights : jax.Array
        weights of the nodes
    likelihood : jax.Array
        likelihood of data-point i from node j.
        must be of shape (n_data x n_nodes)

    Returns
    -------
    gradient of log marginal likelihood: jax.Array
    """
    model = likelihood @ weights
    grad = jnp.mean(likelihood / model[:, jnp.newaxis], axis=0)
    return grad


def _update_weights(
    weights: Float[Array, " n_nodes"], grad: Float[Array, " n_nodes"]
) -> Float[Array, " n_nodes"]:
    """
    Updates weights according to multiplicative gradient algorithm.
    NOTE: this update is positive and sums to 1 without normalization.

    Parameters
    ----------
    weights : jax.Array
        weights of the nodes
    grad : jax.Array
        gradient of weights, same shape as weights

    Returns
    -------
    updated weights: jax.Array
    """
    return weights * grad


def _scaled_gap(grad, weights, scale):
    """
    Find maximum index of gradient vec, only at nonzero indices of weights, and rescale.
    This gap is a proxy for convergence, common for convex objectives.

    Parameters
    ----------
    grad : jax.Array
        gradient of weights, same shape as weights
    weights : jax.Array
        weights of the nodes
    scale : float
        scaling factor, so that the gap at initial iterate is 1.

    Returns
    -------
    scaled gap: float
    """
    grad = jnp.asarray(jnp.where(weights > 0, grad, 0))
    return (jnp.amax(grad) - 1) / scale
