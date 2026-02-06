import equinox as eqx
from cryospax import RelionParticleDataset, RelionParticleParameterFile

from cryojax_eo.ensemble_optimization import HierarchicalSO3GridSearch
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
        RelionParticleParameterFile(
            sample_path_to_starfile, options=dict(broadcasts_image_config=True)
        ),
        sample_path_to_relion_project,
    )
    pose_search = HierarchicalSO3GridSearch(base_grid_res=1, n_rounds=5, n_candidates=40)
    stack = relion_dataset[0]
    pose = pose_search(
        gmm_volume,
        stack["images"],
        stack["parameters"]["image_config"],
        stack["parameters"]["transfer_theory"],
    )
    eqx.tree_equal(pose, stack["parameters"]["pose"], rtol=1e-2)

    return
