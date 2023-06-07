import numpy as np
from sklearn.cluster import KMeans


def calc_population_models(
    models: np.ndarray, weights: np.ndarray, data: np.ndarray, n_clusters: int
) -> None:
    kmeans = KMeans(n_clusters=n_clusters, n_init=4, random_state=0)
    kmeans.fit(data)

    models_cluster_index = kmeans.predict(models)

    for i in range(3):
        print(
            f"Obtained: {np.sum(weights[models_cluster_index==i]):.3f}, "
            + f"Data: {np.sum(kmeans.labels_ == i) / data.shape[0]:.3f}"
        )

    return
