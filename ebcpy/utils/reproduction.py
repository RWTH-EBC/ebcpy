"""
This module contains scripts to extract information
out of simulation / programming based research and
enable a reproduction of the results at a later stage.
"""
import json
import pathlib
import sys
import platform
import os
import logging
from typing import List, Union
import zipfile
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ReproductionFile:
    """
    Data-class for a text-file which will be written to te zip.

    Arguments:
        filename: str
            Name of the file in the zip. Can be a relative path.
        content: str
            Content of the text file
    """
    filename: str
    content: str


@dataclass
class CopyFile:
    """
    Data-class for information on a file
    which will be copied to the zip

    :param str filename:
        Name of the file in the zip. Can be a relative path.
    :param pathlib.Path sourcepath:
        Path on the current machine where the file to copy
        is located
    :param bool remove:
        If True, the file will be moved instead of just copied.
    """
    filename: str
    sourcepath: pathlib.Path
    remove: bool


def save_reproduction_archive(
        title: str,
        path: pathlib.Path = None,
        log_message: str = None,
        files: List[Union[ReproductionFile, CopyFile]] = None,
        file: pathlib.Path = None,
        search_on_pypi: bool = False
):
    """
    Function to save a reproduction archive which contains
    files to reproduce any simulation/software based study.

    :param str title:


    """
    _py_requirements_name = "python/requirements.txt"
    if path is None:
        path = os.getcwd()
    if file is None:
        file = pathlib.Path(sys.modules['__main__'].__file__).absolute()
    if title is None:
        title = file.name.replace(".py", "")
    path = pathlib.Path(path)
    os.makedirs(path, exist_ok=True)
    if files is None:
        files = []
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Start with the file currently running:
    file_running = pathlib.Path(file).absolute()
    files.append(ReproductionFile(
        filename=file_running.name,
        content=file_running.read_text()
    ))
    # Check if it's a git-repo:
    for _dir_path in [file_running] + list(file_running.parents):
        repo_info = get_git_information(
            path=_dir_path,
            software_type="study_repository"
        )
        if repo_info is not None:  # That means it's a repo
            files.extend(repo_info.pop("difference_files", []))
            files.append(ReproductionFile(
                filename="study_repository/repo_info.txt",
                content=json.dumps(repo_info, indent=2)
            ))
            break
    # Get log
    if log_message is None:
        log_message = input("Please enter the specifications / log for this study: ")
        if not log_message:
            log_message = "The user was to lazy to pass any useful information on " \
                  "what made this research study different to others."
    with open(path.joinpath(f"Study_Log_{title}.txt"), "a+") as f:
        f.write(f"{current_time}: {log_message}\n")

    # General info
    files.append(ReproductionFile(
        filename="Information_to_reproduce.txt",
        content=_get_general_information(
            title=title,
            log_message=log_message,
            current_time=current_time
        )
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
        title=title
    )
    files.append(ReproductionFile(
        filename="python/Reproduce_python_environment.txt",
        content=py_repro,
    ))

    zip_file_name = path.joinpath(
        f"{current_time}_{title}.zip"
    )
    with zipfile.ZipFile(zip_file_name, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # Save all result files:
        for file in files:
            if isinstance(file, str):
                if os.path.exists(file):
                    zip_file.write(file, f"Results/{pathlib.Path(file).name}")
                logger.error("Given file '%s' is a string but "
                             "not an existing file. Skipping...", file)
            if isinstance(file, ReproductionFile):
                zip_file.writestr(file.filename, file.content)
            elif isinstance(file, CopyFile):
                zip_file.write(file.sourcepath, file.filename)
                if file.remove:
                    os.remove(file.sourcepath)
            else:
                raise TypeError(
                    f"Given file '{file}' has no "
                    f"valid type. Type is '{type(file)}'")
    return zip_file_name


def get_git_information(
        path: pathlib.Path,
        software_type: str,
        name: str = None
):
    try:
        from git import Repo, InvalidGitRepositoryError, RemoteReference
    except ImportError as err:
        raise ImportError(
            "Could not save data for reproduction, install GitPython using "
            "`pip install GitPython`: " + str(err)
        )
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
            filename=f"{software_type}/WARNING_GIT_DIFFERENCE_{name}_to_local_head.txt",
            content=diff_last_cmt,
        ))
    # Check if pushed to remote
    if not repo.git.branch("-r", contains=commit_hex):
        data["difference_files"].append(ReproductionFile(
            filename=f"{software_type}/WARNING_GIT_DIFFERENCE_{name}_to_remote_main.txt",
            content=diff_remote_main,
        ))
        data["commit"] = remote_main_cmt
    return data


def _get_general_information(title: str, log_message: str, current_time:str):
    """
    Function to save the general information of the study.
    Time, machine information, and an intro on how to reproduce
    the study is given.
    """

    info_header = f"""This folder contains information necessary to reproduce the python based research study named '{title}'.
Reason the user performed this study:
"%s"

To reproduce, make sure you have installed the following programs:
- Anaconda
- Dymola (If a folder named Dymola exists in this zip)

Run the lines in the file 'python/reproduce_python_environment.txt' in a shell with the PATH variable pointing to anaconda (or in Anaconda Prompt).
After execution, make sure to check for any differences in git-based python code.
These files are included in this folder and are named e.g. "WARNING_GIT_DIFFERENCE_some_package".
If this happens, make sure to change the files in the git-based python packages after installation.
For future use, be sure to commit and push your changes before running any research study.
""" % log_message
    _data = {
        "Time": current_time,
        "Author": os.getlogin(),
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
        repo_info = get_git_information(
            path=package.location,
            name=package.key,
            software_type="python"
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


def _get_python_reproduction(title: str):
    """
    Get the content of a script to reproduce the python
    environment used for the study.
    """
    _v = sys.version_info
    py_version = ".".join([str(_v.major), str(_v.minor), str(_v.micro)])
    env_name = f"py_{title}"
    py_reproduce_content = [
        f"conda create -n {env_name} python={py_version} -y",
        f"conda activate {env_name}",
        f"pip install --upgrade pip",
        f"pip install -r requirements.txt",
    ]
    return "\n".join(py_reproduce_content)


if __name__ == '__main__':
    save_reproduction_archive(
        title="my_study",
        path=r"D:\00_temp\reproduction",
    )