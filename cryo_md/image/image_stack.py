"""
Class for storing images and their parameters.
"""
import os
import starfile
import mrcfile
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils import data
import jax.numpy as jnp
from jax.tree_util import tree_map


def numpy_collate(batch):
    return tree_map(np.asarray, data.default_collate(batch))


class NumpyLoader(data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )


class FlattenAndCast(object):
    def __call__(self, pic):
        return np.ravel(np.array(pic, dtype=jnp.float32))


class RelionDataLoader(Dataset):
    def __init__(self, data_path: str, name_star_file: str, res: float):
        self.data_path = data_path
        self.name_star_file = name_star_file
        self.res = res

        self.df = starfile.read(os.path.join(self.data_path, self.name_star_file))
        self.validate_starfile()
        self.num_projs = len(self.df["particles"])
        self.compute_grids()

    def validate_starfile(self):
        if "particles" not in self.df.keys():
            raise Exception("Missing particles in starfile")

        if "optics" not in self.df.keys():
            raise Exception("Missing optics in starfile")

        self.validate_params()

        return

    def validate_params(self):
        req_keys = [
            "rlnDefocusU",
            "rlnDefocusV",
            "rlnDefocusAngle",
            "rlnPhaseShift",
            "rlnOriginXAngst",
            "rlnOriginYAngst",
            "rlnAngleRot",
            "rlnAngleTilt",
            "rlnAnglePsi",
            "rlnNoiseVariance",
            "rlnImageName",
        ]

        missing_keys = []

        for key in req_keys:
            if key not in self.df["particles"].keys():
                missing_keys.append(key)

        if len(missing_keys) > 0:
            raise Exception(
                "Missing required keys in starfile: {}".format(missing_keys)
            )

        return

    def get_df_optics_params(self):
        return (
            self.df["optics"]["rlnImageSize"][0],
            self.df["optics"]["rlnVoltage"][0],
            self.df["optics"]["rlnImagePixelSize"][0],
            self.df["optics"]["rlnSphericalAberration"][0],
            self.df["optics"]["rlnAmplitudeContrast"][0],
        )

    def __len__(self):
        return self.num_projs

    def compute_grids(self):
        box_size = self.df["optics"]["rlnImageSize"][0]
        pixel_size = self.df["optics"]["rlnImagePixelSize"][0]

        values = np.linspace(-0.5, 0.5, box_size + 1)[:-1]
        self.proj_grid = values * pixel_size * box_size

        fx2, fy2 = np.meshgrid(-values, values, indexing="xy")
        self.ctf_grid = np.stack((fx2.ravel(), fy2.ravel()), axis=1) / pixel_size

    def get_ctf_params(self, particle):
        box_size, volt, pixel_size, cs, amp_contrast = self.get_df_optics_params()
        volt = volt * 1000.0
        cs = cs * 1e7
        lam = 12.2639 / np.sqrt(volt + 0.97845e-6 * volt**2)

        ctf_params = np.zeros(9)
        ctf_params[0] = np.array(particle["rlnDefocusU"])
        ctf_params[1] = np.array(particle["rlnDefocusV"])
        ctf_params[2] = np.array(np.radians(particle["rlnDefocusAngle"]))
        ctf_params[3] = np.array(np.radians(particle["rlnPhaseShift"]))
        ctf_params[4] = amp_contrast
        ctf_params[5] = cs

        if "rlnCtfBfactor" in particle.keys():
            ctf_params[6] = np.array(particle["rlnCtfBfactor"])
        else:
            ctf_params[6] = 0.0

        if "rlnCtfScalefactor" in particle.keys():
            ctf_params[7] = np.array(particle["rlnCtfScalefactor"])
        else:
            ctf_params[7] = 1.0

        ctf_params[8] = lam

        return ctf_params

    def __getitem__(self, idx):
        particle = self.df["particles"].iloc[idx]
        try:
            # Load particle image from mrcs file
            imgnamedf = particle["rlnImageName"].split("@")
            mrc_path = os.path.join(self.data_path, imgnamedf[1])
            pidx = int(imgnamedf[0]) - 1
            with mrcfile.mmap(mrc_path, mode="r", permissive=True) as mrc:
                proj = mrc.data[pidx]
        except:
            raise Exception("Error loading image from mrcs file")

        # Read relion orientations and shifts
        pose_params = np.zeros(5)
        pose_params[0] = np.array(particle["rlnOriginXAngst"])
        pose_params[1] = np.array(particle["rlnOriginYAngst"])
        
        pose_params[2:] = np.radians(
            np.stack(
                [
                    particle["rlnAngleRot"],
                    particle["rlnAngleTilt"],
                    particle["rlnAnglePsi"],
                ]
            )
        )

        noise_var = particle["rlnNoiseVariance"]

        # Generate CTF from relion paramaters
        ctf_params = self.get_ctf_params(particle)
        img_params = {
            "proj": proj,
            "pose_params": pose_params,
            "ctf_params": ctf_params,
            "noise_var": noise_var,
            "idx": idx,
        }

        return img_params


def load_starfile(data_path: str, name_star_file: str, res: float, batch_size: int, shuffle: bool = True):
    """
    Load relion starfile into np dataloader adapted to numpy arrays

    Parameters
    ----------
    data_path : str
        Path to starfile and mrcs files
    name_star_file : str
        Name of starfile
    batch_size : int
        Batch size for dataloader

    Returns
    -------
    dataloader : np dataloader
        Dataloader with numpy arrays
    """
    image_stack = RelionDataLoader(
        data_path=data_path,
        name_star_file=name_star_file,
        res=res,
    )

    dataloader = NumpyLoader(
        image_stack,
        shuffle=shuffle,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=0,
    )

    return dataloader
