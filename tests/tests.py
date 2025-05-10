import unittest

import jax
import numpy as np
from jax.config import config as jax_config
from parameterized import parameterized

from cryojax_ensemble_refinement.lklhood_and_grads import (
    calc_lklhood,
    calc_lklhood_grad_strucs,
    calc_lklhood_grad_weights,
)


jax_config.update("jax_enable_x64", True)


class TestLklhoodGradient(unittest.TestCase):
    @parameterized.expand([(10000, 3, 2), (10000, 100, 2), (10000, 100, 20)])
    def test_lklhood_gradients(
        self, n_data_points: int, n_models: int, n_dims: int
    ) -> None:
        np.random.seed(1234)

        data = np.random.randn(n_data_points, n_dims)
        models = np.random.randn(n_models, n_dims)
        weights = np.abs(np.random.randn(n_models))
        sigma = np.random.rand() * 3

        grad_jax = jax.jit(jax.grad(calc_lklhood, argnums=[0, 1]))

        grad_analytic_structs = calc_lklhood_grad_strucs(models, weights, data, sigma)
        grad_analytic_weights = calc_lklhood_grad_weights(models, weights, data, sigma)

        grad_numeric = grad_jax(models, weights, data, sigma)

        np.testing.assert_array_almost_equal(
            grad_numeric[0], grad_analytic_structs, decimal=12
        )
        np.testing.assert_array_almost_equal(
            grad_numeric[1], grad_analytic_weights, decimal=11
        )

        return
