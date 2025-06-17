import jax.numpy as jnp
import mdtraj
from jaxtyping import Array, Int


def get_atom_indices_from_pdb(
    selection_string: str,
    pdb_file: str,
) -> Int[Array, " n_selected_atoms"]:
    """
    Get the atom indices from a selection string.

    **Arguments:**
        selection_string: The selection string to use to selection_string
        atoms in mdtraj format.
        pdb_file: The path to the PDB file.
    """
    atom_indices = jnp.array(mdtraj.load(pdb_file).topology.select(selection_string))
    if len(atom_indices) == 0:
        raise ValueError(
            f"Selection string '{selection_string}'"
            + "did not match any atoms in the topology."
        )
    return atom_indices
