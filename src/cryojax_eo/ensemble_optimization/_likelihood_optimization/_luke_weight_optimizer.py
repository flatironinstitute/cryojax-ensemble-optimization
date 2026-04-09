import logging

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray


def _train_test_split(
    key: PRNGKeyArray,
    log_likelihood_matrix: Float[Array, "n_data n_nodes"],
    train_percentage: float = 0.8,
) -> tuple[
    Float[Array, "num_train n_nodes"],
    Float[Array, "num_test n_nodes"],
    Int[Array, " num_train"],
    Int[Array, " num_test"],
]:
    """
    Splits log likelihood into two sets, based on rows (images / data points)

    ** Arguments:**

    `key`:
        Key from jax.random.key(seed)
    `log_likelihood_matrix`:
        log-likelihood matrix computed from `n_data` data points and `n_nodes` nodes.
        Where nodes are representations of atomic structures (atomic models or volumes).
    `train_percentage:
        Split percentage of data by rows, by default 0.8

    ** Returns:**

    `train`:
        Training split
    `test`:
        Test split
    `train_idx`:
        Indices of original array used for train split
    `test_idx`:
        Indices of original array used for test split
    """
    n_data = log_likelihood_matrix.shape[0]
    split_size = int(jnp.ceil(train_percentage * log_likelihood_matrix.shape[0]))
    train_idx = jax.random.choice(key, n_data, (split_size,), replace=False)
    test_idx = jnp.setdiff1d(jnp.arange(n_data), train_idx)
    train = log_likelihood_matrix[train_idx, :]
    test = log_likelihood_matrix[test_idx, :]
    return train, test, train_idx, test_idx


@jax.jit
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


@jax.jit
def compute_grad(
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


@jax.jit
def _compute_loss(
    weights: Float[Array, " n_nodes"], likelihood: Float[Array, "n_data n_nodes"]
) -> Float[Array, ""]:
    """
    Computes negative marginal log likelihood loss

    Parameters
    ----------
    weights : jax.Array
         weights of the nodes
    likelihood : jax.Array
        likelihood of data-point i from node j.
        must be of shape (n_data x n_nodes)

    Returns
    -------
    negative log likelihood: float
    """
    return -jnp.mean(jnp.log(likelihood @ weights))


@jax.jit
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


def _update_info(
    weights: Float[Array, " n_nodes"], likelihood: Float[Array, "n_data n_nodes"]
) -> Float[Array, ""]:
    """
    For computing info/diagnostics of weights during iterations of mult. grad.
    NOTE: For now, just loss computed, but could add other things here.

    Parameters
    ----------
    weights : jax.Array
         weights of the nodes
    likelihood : jax.Array
        likelihood of data-point i from node j.
        must be of shape (n_data x n_nodes)

    Returns
    -------
    loss: scalar
    other diagnostics to track...
    """
    # TODO: other jit-complied stats will go in here
    loss = _compute_loss(weights, likelihood)
    return loss


@jax.jit
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


# TODO: behavior for train_test versus gradient gap
def multiplicative_gradient(
    log_likelihood: Float[Array, "n_data n_nodes"],
    tol: float = 1e-2,
    max_iterations: int = 10000,
    weights_frequency: int = 0,
    train_test_key: PRNGKeyArray | None = None,
    train_test: bool = False,
    diagnostic: bool = False,
):
    """
    optimizes the weights with the multiplicative gradient method.

    ** Arguments:**

    - `log_likelihood`:
        Likelihood matrix computed from `n_data` data points and `n_nodes` nodes.
        Where nodes are representations of atomic structures (atomic models or volumes).
    - `tol`:
        Tolerance for the stopping criteria
    - `max_iterations`:
        Maximum iterations if stopping criteria isn't met
    - `weights_frequency`:
        If larger than 0, weights are saved at every weights_frequency iterations
    - `train_test_key`:
        key for splitting into train, split for train test procedure
    - `train_test`:
        If true, a stopping index based on train test procedure will be picked,
        then compared with the gap stopping criteria
    - `diagnostic`:
        if true, method will go to max_iterations, returning max iteration weights.
        This can be used to diagnose how overfit the max iterations are compared to the
        weights from train_test or the gap tolerance.

    ** Returns:**

    - `weights:
        optimized weights for the nodes, based on the multiplicative gradient method.
    - `info`:
        dictionary containing diagnostics and info tracked during optimization, including:
        - `losses`: list of losses at each iteration
        - `gaps`: list of gaps at each iteration
        - `weights`: list of weights at each weights_frequency iterations
            (if weights_frequency > 0)
        - `train_test_idx`: stopping index from train test procedure
            (if train_test is True)
        - `weights_train_test`: weights at stopping index from train test procedure
            (if train_test is True)
        - `weights_gap`: weights at stopping index from gap tolerance criterion
            (if gap criterion is reached)
        - `gap_idx`: stopping index from gap tolerance criterion
            (if gap criterion is reached)
    """

    _, n_nodes = log_likelihood.shape
    assert max_iterations > 0, "max_iterations must be positive"

    # Initialize weights
    weights = (1 / n_nodes) * jnp.ones(n_nodes)

    # Convert log likelihood to likelihood via "soft-max"-ish operation
    likelihood = _normalize_log_likelihood_to_likelihood(log_likelihood)

    # Initialize info tracked
    info = {"losses": [], "gaps": [], "weights": []}

    # Initialize scaling for gap stopping criteria
    gap_scale = _scaled_gap(compute_grad(weights, likelihood), weights, scale=1.0)

    # Initialize stopping criteria checks.
    # particularly if not doing train_test, treat this as reached already
    reached_gap = False
    reached_train_test = not train_test

    # Do train test index picking
    if train_test:
        logging.info("Getting train test stopping index")
        train_test_idx = _multiplicative_gradient_train_test(
            train_test_key,
            log_likelihood,
            wait_time=2,
            max_iterations=max_iterations,
        )
        info["train_test_idx"] = train_test_idx
        logging.info(f"Validation loss increases at idx: {train_test_idx}")

    final_idx = 0
    for k in range(max_iterations):
        # Update info
        loss = _update_info(weights, likelihood)
        info["losses"].append(loss)
        # info["your_favorite_stat"].append(...)

        # Check if saving weights
        if weights_frequency > 0 and k % weights_frequency == 0:
            info["weights"].append(weights)

        # Update grad
        grad = compute_grad(weights, likelihood)

        # Check stopping criterions
        gap = _scaled_gap(grad, weights, gap_scale)
        info["gaps"].append(gap)

        # Check current gap against tolerance
        if not reached_gap and gap < tol:
            info["gap_idx"] = k
            info["weights_gap"] = weights
            reached_gap = True
            logging.info(f"reached gap tolerance, at idx: {k}")
            logging.info(f"gap: {gap}")

        # Check current index against the train_test stopping index
        if train_test:
            if k == train_test_idx:
                info["weights_train_test"] = weights
                reached_train_test = True

        # Check if all stopping criteria met
        if reached_train_test and reached_gap and not diagnostic:
            logging.info(f"exiting! At iteration: {k}")
            break

        # Update weights
        weights = _update_weights(weights, grad)
        final_idx += 1

    # Collect info in array format, and save weights and corresponding
    # indices if requested
    info["final_idx"] = final_idx
    info["losses"] = jnp.stack(info["losses"])
    info["gaps"] = jnp.stack(info["gaps"])
    if weights_frequency > 0:
        info["weights"] = jnp.stack(info["weights"])
        info["weights_idx"] = jnp.arange(len(info["weights"])) * weights_frequency

    if not reached_gap:
        logging.info("Terminated at max iters: ")
        logging.info(
            "Returned weights & 'info[weights_gap']' are weights at max_iterations"
        )
        info["weights_gap"] = weights
        info["gap_idx"] = final_idx
    return weights, info


def _multiplicative_gradient_train_test(
    key,
    log_likelihood,
    wait_time=2,
    max_iterations=10000,
    train_pct=0.8,
    smooth_val=0.3,
):
    """
    Rudimentary train test split for finding a stopping index
    of multiplicative gradient.

    Parameters
    ----------
    key: jax.PRNGKey
        key for splitting into train, split for train test procedure
    log_likelihood: jax.Array
        log-likelihood of generating data point i from node j
    wait_time: int
        how many increases in validation loss before stopping
    max_iterations: int
        max iterations if stopping criteria isn't met
    train_pct: float
        percentage of dataset used for training data
    smooth_val: float
        smoothing parameter for exponential smoothing of validation loss
    Returns
    -------
    stopping_idx: int
    """
    n_data, n_nodes = log_likelihood.shape

    log_likelihood_train, log_likelihood_test, _, _ = _train_test_split(
        key, log_likelihood, train_pct
    )

    likelihood_train = _normalize_log_likelihood_to_likelihood(log_likelihood_train)
    likelihood_test = _normalize_log_likelihood_to_likelihood(log_likelihood_test)

    # Initialize weights
    weights = (1 / n_nodes) * jnp.ones(n_nodes)

    # Initialize train test procedure
    count = 0
    smoothed_val_loss = _compute_loss(weights, likelihood_test)
    n_iter = 0
    for k in range(max_iterations):
        # Update grad
        grad = compute_grad(weights, likelihood_train)

        # Update weights
        weights_new = _update_weights(weights, grad)

        # Update smoothed_loss
        val_loss_new = _compute_loss(weights_new, likelihood_test)

        # Smooth, if iterated past the soft assignment weights
        if k > 1:
            smoothed_val_loss_new = (smooth_val) * val_loss_new + (
                1 - smooth_val
            ) * smoothed_val_loss
        else:
            smoothed_val_loss_new = val_loss_new

        # Check stopping criterion: increase in (smoothed) validation loss
        val_losses_diff = smoothed_val_loss_new - smoothed_val_loss
        if val_losses_diff > 0:
            count += 1
        if count >= wait_time:
            break
        weights = weights_new
        smoothed_val_loss = smoothed_val_loss_new
        n_iter += 1

    if n_iter == max_iterations - 1:
        logging.info("NOTE: Train-test stopping criterion not reached.")
    return n_iter
