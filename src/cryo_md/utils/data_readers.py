def parse_initial_models_(
    directory_path: str, n_models: int, ref_universe: mda.Universe, filter: str
):
    if filter == "name CA":
        struct_info = jnp.array(
            pdb_parser(f"{directory_path}/init_system_0.pdb", mode="resid")
        )

    elif filter == "not name H*":
        struct_info = jnp.array(
            pdb_parser(f"{directory_path}/init_system_0.pdb", mode="all_atom")
        )

    else:
        raise NotImplementedError(
            "Only CA and all-atom models are supported at the moment."
        )

    opt_models = np.zeros(
        (n_models, *ref_universe.select_atoms(filter).atoms.positions.T.shape)
    )

    for i in range(n_models):
        univ_system = mda.Universe(
            f"{directory_path}/init_system_{i}.pdb",
        )

        univ_system.atoms.write(f"{directory_path}/curr_system_{i}.pdb")

        univ_prot = univ_system.select_atoms("protein")
        align.alignto(univ_prot, ref_universe, select=filter, match_atoms=True)
        opt_models[i] = univ_prot.select_atoms(filter).atoms.positions.T

    opt_models = jnp.array(opt_models)

    atom_indices = (
        mda.Universe(f"{directory_path}/init_system_0.pdb")
        .select_atoms(f"protein and {filter}")
        .atoms.indices
    )

    atom_indices = jnp.array(atom_indices)

    unit_cell = mda.Universe(f"{directory_path}/init_system_0.pdb").atoms.dimensions
    unit_cell = jnp.array(unit_cell)

    return opt_models, struct_info, atom_indices, unit_cell
