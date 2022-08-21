"""
This module contains scripts to extract information
out of simulation / programming based research and
enable a reproduction of the results at a later stage.

Features:
- Reproduce python environment
- Reproduce git-repos
- Simulation with Dymola
    - SaveTotalModel
    - Save FMU
    - Git Logger
"""

import pathlib
import sys
import platform
import os
import logging
from typing import List
import zipfile
from datetime import datetime
from dataclasses import dataclass
try:
    from git import Repo, InvalidGitRepositoryError, RemoteReference
except ImportError as err:
    raise ImportError(
        "Could not save data for reproduction as "
        "optional dependency is not installed: " + str(err)
    )
logger = logging.getLogger(__name__)


@dataclass
class ReproductionFile:
    filename: str
    content: str


@dataclass
class CopyFile:
    filename: str
    sourcepath: pathlib.Path
    remove: bool


def save_reproduction(
        file: pathlib.Path = None,
        title: str = None,
        save_path: pathlib.Path = None,
        files: List[ReproductionFile] = None,
        search_on_pypi: bool = False
):
    _py_requirements_name = "requirements.txt"
    if save_path is None:
        save_path = os.getcwd()
    if file is None:
        file = pathlib.Path(sys.modules['__main__'].__file__).absolute()
    if title is None:
        title = file.name.replace(".py", "")
    save_path = pathlib.Path(save_path)
    os.makedirs(save_path, exist_ok=True)
    if files is None:
        files = []

    # Start with the file currently running:
    file_running = pathlib.Path(file).absolute()
    files.append(ReproductionFile(
        filename=file_running.name,
        content=file_running.read_text()
    ))
    # General info
    files.append(ReproductionFile(
        filename="general_information.txt",
        content=_get_general_information()
    ))

    # Python-Reproduction:
    py_requirements_content, diff_files = _get_python_package_information(
        search_on_pypi=search_on_pypi
    )
    files.append(ReproductionFile(
        filename=_py_requirements_name,
        content=py_requirements_content,
    ))
    files.extend(diff_files)

    py_repro = _get_python_reproduction(
        requirements_name=_py_requirements_name,
        title=title
    )
    files.append(ReproductionFile(
        filename="reproduce_python_environment.bat",
        content=py_repro,
    ))

    zip_file_name = save_path.joinpath(
        f"{title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    )
    with zipfile.ZipFile(zip_file_name, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # Save all result files:
        for file in files:
            if isinstance(file, ReproductionFile):
                zip_file.writestr(file.filename, file.content)
            elif isinstance(file, CopyFile):
                zip_file.write(file.sourcepath, file.filename)
                if file.remove:
                    os.remove(file)
            else:
                raise TypeError(
                    f"Given file '{file}' has no "
                    f"valid type. Type is '{type(file)}'")
    return zip_file_name


def _get_general_information():
    """
    Function to save the general information of the study.
    Time, machine information, and an intro on how to reproduce
    the study is given.
    """
    info_header = """This folder contains information necessary to reproduce a python based research study.
To reproduce, make sure you have installed the following programs:
- Anaconda
Execute the file 'reproduce_python.bat' in a shell with the PATH variable pointing to anaconda (or in Anaconda Prompt).
After execution, make sure to check for any differences in git-based python code.
These files are included in this folder and are named e.g. "WARNING_GIT_DIFFERENCE_some_package".
If this happens, make sure to change the files in the git-based python packages after installation.
For future use, be sure to commit and push your changes before running any research study.
"""
    _data = {
        "Time": datetime.now(),
        "Machine": platform.machine(),
        "Version": platform.version(),
        "Platform": platform.platform(),
        "System": platform.system(),
        "Processor": platform.processor(),
    }
    _content_lines = [
        info_header + "\n",
        "General system information of performed study:",
    ] + [f"{k}: {v}" for k, v in _data.items()]
    return "\n".join(_content_lines)


def _get_python_package_information(search_on_pypi: bool):
    """
    Function to get the content of python packages installed
    as a requirement.txt format content.
    """
    try:
        from pip._internal.utils.misc import get_installed_distributions
    except ImportError:  # pip<10
        from pip import get_installed_distributions
    installed_packages = get_installed_distributions()
    diff_paths = []
    requirement_txt_content = []
    for package in installed_packages:
        repo_info = _get_git_information(
            path=package.location,
            name=package.key
        )
        if repo_info is None:
            # Check if in python path:
            requirement_txt_content.append(
                f"{package.key}=={package.version}"
            )
            if search_on_pypi:
                from pypisearch.search import Search
                res = Search(package.key).result
                if not res:
                    raise ModuleNotFoundError(
                        "Package '%s' is neither a git "
                        "repo nor a package on pypi. "
                        "Won't be able to reproduce it!",
                        package.key
                    )
        else:
            cmt_sha = repo_info["commit"]
            requirement_txt_content.append(
                f"git+{repo_info['url']}.git@{cmt_sha}#egg={package.key}"
            )
            diff_paths.extend(repo_info["difference_files"])
    return "\n".join(requirement_txt_content), diff_paths


def _get_python_reproduction(requirements_name: str, title: str):
    """
    Get the content of a script to reproduce the python
    environment used for the study.
    """
    _v = sys.version_info
    py_version = ".".join([str(_v.major), str(_v.minor), str(_v.micro)])
    env_name = f"py_reproduce_{title}"
    py_reproduce_content = [
        f"conda create -n {env_name} python={py_version} -y",
        f"conda activate {env_name}",
        f"pip install --upgrade pip",
        f"pip install -r {requirements_name}",
        f"conda deactivate",
    ]
    return "\n".join(py_reproduce_content)


def _get_git_information(
        path: pathlib.Path,
        name: str = None,
):
    try:
        repo = Repo(path)
    except InvalidGitRepositoryError:
        return
    commit = repo.head.commit
    commit_hex = commit.hexsha
    diff_last_cmt = repo.git.diff(commit)
    diff_remote_main = ""
    remote_main_cmt = ""
    for ref in repo.references:
        if isinstance(ref, RemoteReference) and ref.name in ['origin/master', 'origin/main']:
            diff_remote_main = repo.git.diff(ref.commit)
            remote_main_cmt = ref.commit.hexsha
            break
    data = {
        "url": next(repo.remotes[0].urls),
        "commit": commit_hex,
        "difference_files": []
    }

    if name is None:
        # Get last part of url
        name = data["url"].split("/")[-1].replace(".git", "")
    # Check new files
    if diff_last_cmt:
        data["difference_files"].append(ReproductionFile(
            filename=f"WARNING_GIT_DIFFERENCE_{name}_to_local_head.txt",
            content=diff_last_cmt,
        ))
    # Check if pushed to remote
    if not repo.git.branch("-r", contains=commit_hex):
        data["difference_files"].append(ReproductionFile(
            filename=f"WARNING_GIT_DIFFERENCE_{name}_to_remote_main.txt",
            content=diff_remote_main,
        ))
        data["commit"] = remote_main_cmt
    return data


if __name__ == '__main__':
    save_reproduction(
        title="my_study",
        save_path=r"D:\00_temp\reproduction",
    )
