import itertools
import os
import gemmi


def read_gemmi_atoms(path, i_model=0, chains=None, clean=False, center=True, pdb_out=""):
    """
    Read atoms, separated by chain, from PDB or CIF file using Gemmi.

    Parameters
    ----------
    path : string
        Path to PDB or mmCIF file.
    i_model : integer
        Optional, default: 0
        Index of the returned model in the Gemmi Structure.
    chains : list of strings
        chains to select, optional.
        If not provided, retrieve atoms from all chains.
    clean : bool
        Optional, default: True
        If True, use Gemmi remove_* methods to clean up structure.
    center : bool
        Optional, default: True
        If True, position the coordinates with the center of mass at the origin.
    pdb_out : string
        Optional, default: None
        If provided, write the optionally cleaned, centered model to PDB or cif.

    Returns
    -------
    atoms : list of list(s) of Gemmi atoms
        Gemmi atom objects associated with each chain
    """
    model = read_gemmi_model(
        path, i_model=i_model, clean=clean, center=center, pdb_out=pdb_out
    )
    return extract_gemmi_atoms(model, chains=chains)


def read_gemmi_model(path, i_model=0, clean=True, center=True, pdb_out=""):
    """
    Use Gemmi library to read PDB or mmCIF files and return a Gemmi model.
    The hierarchy in Gemmi follows:
    Structure - Model - Chain - Residue - Atom

    Parameters
    ----------
    path : string
        Path to PDB or mmCIF file.
    i_model : integer
        Optional, default: 0
        Index of the returned model in the Gemmi Structure.
    clean : bool
        Optional, default: True
        If True, use Gemmi remove_* methods to clean up structure.
    center : bool
        Optional, default: True
        If True, position the coordinates with the center of mass at the origin.
    pdb_out : string
        Optional, default: None
        If provided, write the optionally cleaned, centered model to PDB or cif.

    Returns
    -------
    model: Gemmi Class
        Gemmi model
    """
    if os.path.isfile(path):
        is_pdb = path.lower().endswith(".pdb")
        if is_pdb:
            model = read_gemmi_model_from_pdb(path, i_model, clean)
        else:
            raise ValueError("File format not recognized.")
    else:
        raise OSError("File could not be found.")

    if center:
        center_gemmi_model(model)
    if pdb_out != "":
        write_gemmi_model(pdb_out, model)

    return model


def read_gemmi_model_from_pdb(path, i_model=0, clean=True):
    """
    Read Gemmi Model from PDB file.

    Parameters
    ----------
    path : string
        Path to PDB file.
    i_model : integer
        Optional, default: 0
        Index of the returned model in the Gemmi Structure.
    clean : bool
        Optional, default: True
        If True, use Gemmi remove_* methods to clean up structure.

    Returns
    -------
    model : Gemmi Class
        Gemmi model
    """
    structure = gemmi.read_structure(path)
    if clean:
        structure = clean_gemmi_structure(structure)
    model = structure[i_model]
    return model


def clean_gemmi_structure(structure=None):
    """
    Clean Gemmi Structure, removing alternate conformations, hydrogens,
    waters, ligands, and empty chains.

    Parameters
    ----------
    structure : Gemmi Class
        Gemmi Structure object

    Returns
    -------
    structure : Gemmi Class
        Same object, cleaned up of unnecessary atoms.
    """
    if structure is not None:
        try:
            structure.remove_alternative_conformations()
        except RuntimeError:
            print("No alternative conformations to remove.")
            pass
        try:
            structure.remove_hydrogens()
        except RuntimeError:
            print("No hydrogens to remove.")
            pass
        try:
            structure.remove_waters()
        except RuntimeError:
            print("No waters to remove.")
            pass
        try:
            structure.remove_ligands_and_waters()
        except RuntimeError:
            print("No ligands to remove.")
            pass
        try:
            structure.remove_empty_chains()
        except RuntimeError:
            print("No empty chains to remove.")
            pass

    return structure


def center_gemmi_model(model):
    """
    Translates model so that its center of mass coincides with the origin.

    Parameters
    ----------
    model : Gemmi Class
        Gemmi model
    """
    com = model.calculate_center_of_mass()
    model.transform_pos_and_adp(
        gemmi.Transform(
            gemmi.Mat33(),  # rotation matrix is identity
            gemmi.Vec3(-1 * com.x, -1 * com.y, -1 * com.z),
        )
    )
    return


def write_gemmi_model(path, model=gemmi.Model("model")):
    """
    Write Gemmi model to PDB or mmCIF file.

    Parameters
    ----------
    path : string
        Path to PDB or mmCIF file.
    model : Gemmi Class
        Optional, default: gemmi.Model()
        Gemmi model
    """
    is_pdb = path.lower().endswith(".pdb")
    if not (is_pdb):
        raise ValueError("File format not recognized.")

    structure = gemmi.Structure()
    structure.add_model(model, pos=-1)
    structure.renumber_models()

    if is_pdb:
        structure.write_pdb(path)


def extract_gemmi_atoms(model, chains=None, split_chains=False):
    """
    Extract Gemmi atoms from the input model, separated by chain.

    Parameters
    ----------
    model : Gemmi Class
        Gemmi model
    chains : list of strings
        chains to select, optional.
        If not provided, retrieve atoms from all chains.
    split_chains : bool
        Optional, default: False
        if True, keep the atoms from different chains in separate lists

    Returns
    -------
    atoms : list (or list of list(s)) of Gemmi atoms
        Gemmi atom objects, either concatenated or separated by chain
    """
    if chains is None:
        chains = [ch.name for ch in model]

    atoms = []
    for ch in model:
        if ch.name in chains:
            atoms.append([at for res in ch for at in res])

    if not split_chains:
        atoms = list(itertools.chain.from_iterable(atoms))

    return atoms


def extract_atomic_parameter(atoms, parameter_type, split_chains=False):
    """
    Interpret Gemmi atoms and extract a single parameter type.

    Parameters
    ----------
    atoms : list (of list(s)) of Gemmi atoms
        Gemmi atom objects associated with each chain
    parameter_type : string
        'cartesian_coordinates', 'form_factor_a', or 'form_factor_b'
    split_chains : bool
        Optional, default: False
        if True, keep the atoms from different chains in separate lists

    Returns
    -------
    atomic_parameter : list of floats, or list of lists of floats
        atomic parameter associated with each atom, optionally split by chain
    """
    # if list of Gemmi atoms, convert into a list of list
    if not isinstance(atoms[0], list):
        atoms = [atoms]

    if parameter_type == "cartesian_coordinates":
        atomic_parameter = [at.pos.tolist() for ch in atoms for at in ch]
    elif parameter_type == "form_factor_a":
        atomic_parameter = [at.element.c4322.a for ch in atoms for at in ch]
    elif parameter_type == "form_factor_b":
        atomic_parameter = [at.element.c4322.b for ch in atoms for at in ch]
    elif parameter_type == "B_factor":
        atomic_parameter = [at.b_iso for ch in atoms for at in ch]
    else:
        raise ValueError("Atomic parameter type not recognized.")
        return

    # optionally preserve the list of lists (separated by chain) structure
    if split_chains:
        reshape = [0] + [len(ch) for ch in atoms]
        atomic_parameter = [
            atomic_parameter[reshape[i] : reshape[i] + reshape[i + 1]]
            for i in range(len(reshape) - 1)
        ]

    return atomic_parameter