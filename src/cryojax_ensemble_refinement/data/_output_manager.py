import logging
import os

import h5py
import numpy as np


class OutputManager:
    def __init__(self, file_name: str, steps, models_shape, dtype=np.float64):
        self.file_name = self.check_file(file_name)
        self.file = h5py.File(self.file_name, "w")
        self.dtype = dtype
        self.create_datasets(steps, models_shape)

        return

    def __del__(self):
        self.file.close()
        return

    def close(self):
        self.file.close()
        return

    def check_file(self, file_name: str):
        if os.path.exists(file_name):
            logging.warning("File already exists, creating backup...")
            os.rename(file_name, file_name + ".bak")

        if not file_name.endswith(".h5"):
            logging.warning("File name does not end with .h5, adding extension...")
            file_name += ".h5"

        return file_name

    def create_datasets(self, steps, models_shape):
        self.trajs_positions = self.file.create_dataset(
            "trajs_positions",
            (steps, *models_shape),
            dtype=self.dtype,
        )
        self.trajs_weights = self.file.create_dataset(
            "trajs_weights", (steps, models_shape[0]), dtype=self.dtype
        )
        self.losses = self.file.create_dataset("losses", (steps,), dtype=self.dtype)

        return

    def write(self, positions, weights, loss, step):
        self.trajs_positions[step] = positions
        self.trajs_weights[step] = weights
        self.losses[step] = loss

        self.file.flush()

        return
