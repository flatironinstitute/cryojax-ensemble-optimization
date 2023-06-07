import numpy as np


def data_generator(
    centers: np.ndarray, cov_mats: np.ndarray, points_per_center: np.ndarray
) -> np.ndarray:
    data = np.zeros((points_per_center.sum(), centers.shape[1]))

    for i in range(centers.shape[0]):
        data[
            points_per_center[:i].sum() : points_per_center[: i + 1].sum()
        ] = np.random.multivariate_normal(centers[i], cov_mats, points_per_center[i])

    return data
