import logging
from tqdm import tqdm
from MDAnalysis.analysis import align
import numpy as np
import jax.numpy as jnp

from ._molecular_dynamics.mdaa_simulator import MDSampler
from ._molecular_dynamics.mdcg_simulator import MDCGSampler
from ._optimization.optimizer import WeightOptimizer, PositionOptimizer
from ._data.output_manager import OutputManager


class Pipeline:
    def __init__(self, workflow, config):
        self.check_steps(workflow)
        self.workflow = workflow
        self.config = config
        # self.output_manager = output_manager

        return

    def check_steps(self, workflow):
        logging.info("Reading pipeline workflow...")

        self.workflow_types = []
        for i, step in enumerate(workflow):
            if isinstance(step, MDSampler):
                self.workflow_types.append("MDSampler")
                logging.info(f"Step {i}: MDSampler")

            elif isinstance(step, MDCGSampler):
                self.workflow_types.append("MDCGSampler")
                logging.info(f"Step {i}: MDCGSampler")

            elif isinstance(step, WeightOptimizer):
                self.workflow_types.append("WeightOptimizer")
                logging.info(f"Step {i}: WeightOptimizer")

            elif isinstance(step, PositionOptimizer):
                self.workflow_types.append("PositionOptimizer")
                logging.info(f"Step {i}: PositionOptimizer")

            else:
                logging.error(
                    f"Invalid step {i}, must be MDSampler, WeightOptimizer, or PositionOptimizer"
                )
                raise ValueError(
                    f"Invalid step type: {type(step)}, must be MDSampler, WeightOptimizer, or PositionOptimizer"
                )
        logging.info(f"Loaded workflow with steps: {self.workflow_types}")
        logging.info("Checking if workflow is valid")
        self.workflow_is_valid()
        logging.info("Workflow check complete.")

        return

    def workflow_is_valid(self):
        if all(x in ["MDCGSampler", "MDSampler"] for x in self.workflow_types):
            logging.error("Workflow cannot contain MDCGSampler and MDSampler")
            raise ValueError("Workflow cannot contain MDCGSampler and MDSampler")

        if not any(x in ["MDCGSampler", "MDSampler"] for x in self.workflow_types):
            logging.error(
                "Workflow must contain at least one of MDCGSampler or MDSampler"
            )
            raise ValueError(
                "Workflow must contain at least one of MDCGSampler or MDSampler"
            )

        if "WeightOptimizer" not in self.workflow_types:
            logging.error("Workflow must contain at least one WeightOptimizer")
            raise NotImplementedError(
                "Workflow must contain at least one WeightOptimizer"
            )

        if "PositionOptimizer" not in self.workflow_types:
            logging.error("Workflow must contain at least one PositionOptimizer")
            raise NotImplementedError(
                "Workflow must contain at least one PositionOptimizer"
            )

        for i in range(len(self.workflow_types)):
            logging.info(f"Checking step {i} with type: {self.workflow_types[i]}")
            if i == 0 and self.workflow_types[i] != "WeightOptimizer":
                logging.error("First step must be WeightOptimizer")
                raise NotImplementedError("First step must be WeightOptimizer")

            if i == 1 and self.workflow_types[i] != "PositionOptimizer":
                logging.error("Second step must be PositionOptimizer")
                raise NotImplementedError("Second step must be PositionOptimizer")

            if i > 1 and self.workflow_types[i] not in ["MDSampler", "MDCGSampler"]:
                logging.error(f"Step {i} is {self.workflow_types[i]}")
                logging.error("All steps after second must be MDSampler or MDCGSampler")
                raise NotImplementedError(
                    "All steps after second must be MDSampler or MDCGSampler"
                )

        return

    def prepare_for_run_(
        self,
        n_steps,
        init_universes,
        struct_info,
        mode,
        output_file,
        ref_universe,
        init_weights=None,
    ):
        if "MDSampler" in self.workflow_types:
            if mode not in ["all-atom", "resid"]:
                logging.error(
                    "Invalid mode, must be 'all-atom' or 'resid' when using MDSampler"
                )
                raise ValueError(
                    "Invalid mode, must be 'all-atom' or 'resid' when using MDSampler"
                )

            if mode == "all-atom":
                self.filter = "protein and not name H*"

            elif mode == "resid":
                self.filter = "protein and name CA"

            self.filetype = "pdb"

        if "MDCGSampler" in self.workflow_types:
            if mode in ["all-atom", "resid"]:
                logging.error("Cannot run all-atom optimization with CG MD")
                raise ValueError("Cannot run all-atom optimization with CG MD")

            elif mode == "cg":
                self.filter = "protein"

            self.filetype = "gro"

        self.univ_md = []
        self.univ_pull = []
        self.ref_universe = ref_universe
        self.n_models = len(init_universes)
        self.struct_info = struct_info

        for i in range(len(init_universes)):
            self.univ_md.append(init_universes[i].copy())
            self.univ_pull.append(init_universes[i].copy())

        self.n_steps = n_steps

        models_shape = (
            len(init_universes),
            *init_universes[0].select_atoms("protein").atoms.positions.T.shape,
        )

        self.output_manager = OutputManager(output_file, n_steps, models_shape)

        if init_weights is None:
            self.weights = jnp.ones(self.n_models) / self.n_models

        else:
            self.weights = init_weights.copy()

        self.opt_atom_list = self.univ_md[0].select_atoms(self.filter).atoms.indices
        self.unit_cell = self.univ_md[0].atoms.dimensions

        return

    def run_md_(self, step):
        for i in range(self.n_models):
            self.univ_md[i].atoms.write(f"positions.{self.filetype}")
            self.univ_pull[i].atoms.write(f"ref_positions.{self.filetype}")

            # self.univ_md[i].atoms.write(f"md_init_model_{i}.pdb")
            # self.univ_pull[i].atoms.write(f"md_pull_model_{i}.pdb")
            positions = step.run(
                i, f"ref_positions.{self.filetype}", self.opt_atom_list
            )
            self.univ_md[i].atoms.positions = positions.copy()
            self.univ_md[i].select_atoms(self.filter).atoms.write(
                f"positions_after_md.{self.filetype}"
            )

        return

    def run_wts_opt_(self, step, image_stack):
        positions = np.zeros(
            (
                self.n_models,
                *self.ref_universe.select_atoms(self.filter).atoms.positions.T.shape,
            )
        )

        for i in range(self.n_models):
            dummy_univ = self.univ_md[i].copy()
            align.alignto(
                dummy_univ, self.ref_universe, select=self.filter, match_atoms=True
            )
            positions[i] = dummy_univ.select_atoms(self.filter).atoms.positions.T

        positions = jnp.array(positions)
        self.weights = step.run(
            positions, self.weights, image_stack, self.struct_info, self.config
        )

        return

    def run_pos_opt_(self, step, image_stack):
        positions = np.zeros(
            (
                self.n_models,
                *self.ref_universe.select_atoms(self.filter).atoms.positions.T.shape,
            )
        )

        for i in range(self.n_models):
            dummy_univ = self.univ_md[i].copy()

            # dummy_univ.atoms.write(f"model_before_opt_{i}.pdb")

            align.alignto(
                dummy_univ, self.ref_universe, select=self.filter, match_atoms=True
            )

            positions[i] = dummy_univ.select_atoms(self.filter).atoms.positions.T

        logging.debug(f"Optimized_positions: {positions}")

        positions = jnp.array(positions)
        positions, loss = step.run(
            positions, self.weights, image_stack, self.struct_info, self.config
        )

        logging.debug(f"Optimized_positions: {positions}")

        for i in range(self.n_models):
            dummy_univ = self.univ_md[i].copy()
            align.alignto(
                dummy_univ, self.ref_universe, select=self.filter, match_atoms=True
            )
            dummy_univ.select_atoms(self.filter).atoms.positions = positions[i].T

            align.alignto(
                dummy_univ, self.univ_md[i], select=self.filter, match_atoms=True
            )
            self.univ_pull[i].atoms.positions = dummy_univ.atoms.positions.copy()
            self.univ_pull[i].select_atoms(self.filter).atoms.write(
                f"model_after_opt_{i}.pdb"
            )

        return loss

    def run(self, image_stack):
        logging.info(f"Running pipeline for {self.n_steps} steps...")

        with tqdm(range(self.n_steps), unit="step") as pbar:
            for counter in pbar:
                for step in self.workflow:
                    if isinstance(step, MDSampler) or isinstance(step, MDCGSampler):
                        self.run_md_(step)

                    elif isinstance(step, WeightOptimizer):
                        self.run_wts_opt_(step, image_stack)

                    elif isinstance(step, PositionOptimizer):
                        loss = self.run_pos_opt_(step, image_stack)

                    else:
                        continue

                pbar.set_postfix(loss=loss)
                self.output_manager.write(
                    np.array(
                        [
                            univ.select_atoms("protein").atoms.positions.T
                            for univ in self.univ_md
                        ]
                    ),
                    self.weights,
                    loss,
                    counter,
                )

        logging.info("Pipeline finished.")
        logging.info("Output saved to file.")

        return
