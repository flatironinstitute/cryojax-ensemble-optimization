from cryojax.dataset import RelionParticleDataset, RelionParticleParameterFile

from cryojax_eo.ensemble_optimization import global_SO3_hier_search
from cryojax_eo.io import load_gmm_volume_parametrization


def test_global_SO3_hier_search(
    sample_path_to_pdb1,
    sample_path_to_relion_project,
    sample_path_to_starfile,
):
    gmm_volume = load_gmm_volume_parametrization(
        [sample_path_to_pdb1],
        selection_string="not element H",
    )[0]

    relion_dataset = RelionParticleDataset(
        RelionParticleParameterFile(sample_path_to_starfile),
        sample_path_to_relion_project,
    )

    global_SO3_hier_search(gmm_volume, relion_dataset[0], 1, 5, 40)

    return
