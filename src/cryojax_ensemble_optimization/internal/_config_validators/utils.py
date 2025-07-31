import glob
import os
from pathlib import Path
from typing import List, Union

from natsort import natsorted
from pydantic import (
    DirectoryPath,
    FilePath,
)


def _validate_file_with_type(filename: str, file_type: str) -> str:
    """
    Check if the file is a PDB file.
    """
    assert Path(filename).exists(), f"File {filename} does not exist."
    assert Path(filename).is_file(), f"Path {filename} is not a file."

    if not Path(filename).suffix == f"{file_type}":
        raise ValueError(f"File {filename} is not a {file_type} file.")
    return filename


# Might be useful, and I don't want to figure it out again
# TODO: remove if not needed
def _contains_file_type(
    directory_path: DirectoryPath, file_type: str | List[str]
) -> DirectoryPath:
    if isinstance(file_type, str):
        file_type = [file_type]

    failing_types = []
    for ftype in file_type:
        files_in_directory = glob.glob(os.path.join(directory_path, f"*.{ftype}"))
        if len(files_in_directory) == 0:
            failing_types.append(ftype)

    if len(failing_types) > 0:
        raise ValueError(
            f"Directory {directory_path} does not contain any files "
            + f"of type(s): {', '.join(failing_types)}"
        )

    return directory_path


def _validate_files_with_type(
    path_to_files: Union[str, List[FilePath]], file_types: List[str]
) -> List[str]:
    if isinstance(path_to_files, str):
        if "*" in path_to_files:
            output = [Path(f) for f in natsorted(glob.glob(path_to_files))]
        elif Path(path_to_files).is_file():
            output = [Path(path_to_files)]
        else:
            raise ValueError(
                f"Path {path_to_files} is not a file or does not use * wild card."
            )
    elif isinstance(path_to_files, list):
        output = [Path(f) for f in path_to_files]

    for f in output:
        assert f.exists(), f"{f} does not exist."
        assert f.is_file(), f"{f} is not a file."
        assert f.suffix in file_types, (
            f"{f} is not a valid file type. "
            + f"Valid file types are: {', '.join([f'.{file_types}'])}"
        )
    return [str(f) for f in output]
