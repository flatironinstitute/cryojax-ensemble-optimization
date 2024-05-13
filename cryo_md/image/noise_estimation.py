from aspire.source import RelionSource
from aspire.noise import AnisotropicNoiseEstimator
from aspire.noise import WhiteNoiseEstimator

def estimate_noise():

    path_to_images = "/mnt/home/ccb/public_www/cryoEM_heterogeneity_challenge_2023/cryoem_heterogeneity_challenge_2023_second_dataset/cryoem_heterogeneity_challenge_2023_second_dataset_448x448.star"

    src = RelionSource(
        path_to_images,
        data_folder="",
        pixel_size=1.073,
    )

    src = src.downsample(128)
    aiso_noise_estimator = AnisotropicNoiseEstimator(src)
    src = src.whiten(aiso_noise_estimator)
    noise_estimator = WhiteNoiseEstimator(src)

    noise_variance = noise_estimator.estimate()
    snr = src.estimate_snr()
    
    return noise_variance, snr