import numpy as np

import jax.numpy as jnp
from cryojax.io import read_atoms_from_pdb_or_cif


def pdb_parser_all_atom_(fname: str) -> dict[str, np.ndarray]:
    """
    Parses a pdb file and returns an atomic model of the protein. The atomic model is a 5xN array, where N is the number of atoms in the protein. The first three rows are the x, y, z coordinates of the atoms. The fourth row is the atomic number of the atoms. The fifth row is the variance of the atoms before the resolution is applied.
    Parameters
    ----------
    fname : str
        The path to the pdb file.

    Returns
    -------
    struct_info : dict
        The atomic model of the protein.

    """

    _, atom_identities, b_factors = read_atoms_from_pdb_or_cif(
        fname,
        center=True,
        get_b_factors=True,
        atom_filter="protein and not element H",
        is_assembly=False,
    )

    struct_info = {
        "atom_identities": jnp.array(atom_identities),
        "b_factors": jnp.array(b_factors),
    }

    return struct_info


# def pdb_parser_resid_(fname: str) -> np.array:
#     """
#     Parses a pdb file and returns a coarsed grained atomic model of the protein. The atomic model is a 5xN array, where N is the number of residues in the protein. The first three rows are the x, y, z coordinates of the alpha carbons. The fourth row is the density of the residues, i.e., the total number of electrons. The fifth row is the radius of the residues squared, which we use as the variance of the residues for the forward model.

#     Parameters
#     ----------
#     fname : str
#         The path to the pdb file.

#     Returns
#     -------
#     struct_info : np.array
#         The coarse grained atomic model of the protein.
#     """
#     """
#     resid_radius = {
#         "CYS": 2.75,
#         "PHE": 3.2,
#         "LEU": 3.1,
#         "TRP": 3.4,
#         "VAL": 2.95,
#         "ILE": 3.1,
#         "MET": 3.1,
#         "HIS": 3.05,
#         "TYR": 3.25,
#         "ALA": 2.5,
#         "GLY": 2.25,
#         "PRO": 2.8,
#         "ASN": 2.85,
#         "THR": 2.8,
#         "SER": 2.6,
#         "ARG": 3.3,
#         "GLN": 3.0,
#         "ASP": 2.8,
#         "LYS": 3.2,
#         "GLU": 2.95,
#     }

#     resid_density = {
#         "CYS": 64.0,
#         "PHE": 88.0,
#         "LEU": 72.0,
#         "TRP": 108.0,
#         "VAL": 64.0,
#         "ILE": 72.0,
#         "MET": 80.0,
#         "HIS": 82.0,
#         "TYR": 96.0,
#         "ALA": 48.0,
#         "GLY": 40.0,
#         "PRO": 62.0,
#         "ASN": 66.0,
#         "THR": 64.0,
#         "SER": 56.0,
#         "ARG": 93.0,
#         "GLN": 78.0,
#         "ASP": 59.0,
#         "LYS": 79.0,
#         "GLU": 53.0,
#     }

#     univ = mda.Universe(fname)
#     residues = univ.select_atoms("protein").residues

#     struct_info = np.zeros((2, residues.n_residues))
#     struct_info[0, :] = (
#         np.array([resid_radius[x] for x in residues.resnames]) / np.pi
#     ) ** 2

#     struct_info[1, :] = np.array([resid_density[x] for x in residues.resnames])

#     return struct_info
#     """
#     raise NotImplementedError("This function is not implemented yet.")


# def pdb_parser_cg_(fname: str, config: dict) -> np.array:
#     """
#     Parses a pdb file and returns a coarsed grained atomic model of the protein. The atomic model is a 5xN array, where N is the number of residues in the protein. The first three rows are the x, y, z coordinates of the alpha carbons. The fourth row is the density of the residues, i.e., the total number of electrons. The fifth row is the radius of the residues squared, which we use as the variance of the residues for the forward model.

#     Parameters
#     ----------
#     fname : str
#         The path to the pdb file.

#     Returns
#     -------
#     struct_info : np.array
#         The coarse grained atomic model of the protein.
#     """

#     resid_radius = {
#         "CYS": 2.75,
#         "PHE": 3.2,
#         "LEU": 3.1,
#         "TRP": 3.4,
#         "VAL": 2.95,
#         "ILE": 3.1,
#         "MET": 3.1,
#         "HIS": 3.05,
#         "TYR": 3.25,
#         "ALA": 2.5,
#         "GLY": 2.25,
#         "PRO": 2.8,
#         "ASN": 2.85,
#         "THR": 2.8,
#         "SER": 2.6,
#         "ARG": 3.3,
#         "GLN": 3.0,
#         "ASP": 2.8,
#         "LYS": 3.2,
#         "GLU": 2.95,
#     }

#     resid_density = {
#         "CYS": 64.0,
#         "PHE": 88.0,
#         "LEU": 72.0,
#         "TRP": 108.0,
#         "VAL": 64.0,
#         "ILE": 72.0,
#         "MET": 80.0,
#         "HIS": 82.0,
#         "HSD": 82.0,
#         "TYR": 96.0,
#         "ALA": 48.0,
#         "GLY": 40.0,
#         "PRO": 62.0,
#         "ASN": 66.0,
#         "THR": 64.0,
#         "SER": 56.0,
#         "ARG": 93.0,
#         "GLN": 78.0,
#         "ASP": 59.0,
#         "LYS": 79.0,
#         "GLU": 53.0,
#     }

#     univ = mda.Universe(fname)
#     protein = univ.select_atoms("protein")

#     gauss_var = np.ones(protein.n_atoms) * config["resolution"] ** 2 / (2.0 * np.pi**2)
#     gauss_amp = np.zeros(protein.n_atoms)

#     counter = 0
#     for i in range(protein.n_residues):
#         n_atoms_in_resid = protein.residues[i].atoms.n_atoms
#         for _ in range(n_atoms_in_resid):
#             gauss_amp[counter] = resid_density[protein.residues.resnames[i]]
#             gauss_var[counter] *= (resid_radius[protein.residues.resnames[i]]) ** 2
#             counter += 1

#     gauss_amp = gauss_amp / (2 * np.pi * gauss_var)

#     struct_info = {
#         "gauss_var": jnp.array(gauss_var.reshape(-1, 1)),
#         "gauss_amp": jnp.array(gauss_amp.reshape(-1, 1)),
#     }
#     return struct_info


def pdb_parser(input_file: str, mode="all-atom") -> dict:
    """
    Parses a pdb file and returns an atomic model of the protein. The atomic model is a 5xN array, where N is the number of atoms or residues in the protein. The first three rows are the x, y, z coordinates of the atoms or residues. The fourth row is the atomic number of the atoms or the density of the residues. The fifth row is the variance of the atoms or residues, which is the resolution of the cryo-EM map divided by pi squared.

    Parameters
    ----------
    input_file : str
        The path to the pdb file.
    mode : str
        The mode of the atomic model. Either "resid" or "all-atom". Resid mode returns a coarse grained atomic model of the protein. All atom mode returns an all atom atomic model of the protein.
    resolution : float
        The resolution of the desired cryoEM images.

    Returns
    -------
    struct_info : Array
        The atomic model of the protein.
    """

    # if config["mode"] == "resid":
    #     #struct_info = pdb_parser_resid_(input_file)
    #     raise NotImplementedError("This function is not implemented yet.")

    # elif config["mode"] == "all-atom":
    #     struct_info = pdb_parser_all_atom_(input_file, config)

    # elif config["mode"] == "cg":
    #     #struct_info = pdb_parser_cg_(input_file, config)
    #     raise NotImplementedError("This function is not implemented yet.")

    # else:
    #     raise ValueError("Mode must be either 'cg', 'resid' or 'all-atom'.")

    struct_info = pdb_parser_all_atom_(input_file)
    return struct_info
