import mdtraj
from jaxtyping import ArrayLike, Int


def get_atom_indices_from_pdb(
    select: str,
    pdb_file: str,
) -> Int[ArrayLike, " n_selected_atoms"]:
    """
    Get the atom indices from a selection string.

    **Arguments:**
        select: The selection string to use to select atoms in mdtraj format.
        pdb_file: The path to the PDB file.
    """
    atom_indices = mdtraj.load(pdb_file).topology.select(select)
    if len(atom_indices) == 0:
        raise ValueError(
            f"Selection string '{select}' did not match any atoms in the topology."
        )
    return atom_indices
