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
import shutil
import zipfile
from datetime import datetime
from git import Repo, InvalidGitRepositoryError, RemoteReference

logger = logging.getLogger(__name__)


def save_reproduction(file=None, title=None, save_path=None, files=None):
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
    file_running_save = save_path.joinpath(file_running.name)
    copy_file(src=file_running, dst=file_running_save)
    files.append({"file": file_running_save,
                  "remove": file_running_save != file_running})
    # General info
    s_path = save_general_information(save_path=save_path)
    files.append({"file": s_path, "remove": True})
    # Python-Reproduction:
    requirements_path, diff_files = get_python_packages(save_path=save_path)
    files.append({"file": requirements_path, "remove": True})
    for _file in diff_files:
        files.append({"file": _file, "remove": True})

    s_path = save_python_reproduction(
        requirements_path=requirements_path,
        title=title
    )
    files.append({"file": s_path, "remove": True})

    zip_file = save_to_zip(
        files=[_f["file"] for _f in files],
        title=title,
        save_path=save_path
    )
    # Remove created files:
    for _f in files:
        if _f["remove"]:
            os.remove(_f["file"])
    return zip_file


def save_to_zip(files, title, save_path):
    # Save the study files to a zip for in order to
    # reproduce the results if necessary or better understand them
    zip_file_name = save_path.joinpath(
        f"{title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    )
    with zipfile.ZipFile(zip_file_name, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # Save all result files:
        for file in files:
            # Save all files to directory Result Files
            if os.path.isfile(file):
                zip_file.write(file, pathlib.Path(file).name)
            else:
                raise OSError("Can only save files not folders at current state")
    return zip_file_name


def copy_file(src, dst):
    if not os.path.isfile(src):
        raise OSError("Can only copy files at current state")
    if src != dst:
        shutil.copy(src, dst)


def save_general_information(save_path):
    info_header = """This folder contains information necessary to reproduce a python based research study.
To reproduce, make sure you have installed the following programs:
- Anaconda
Execute the file 'reproduce_python.bat' in a shell with the PATH variable pointing to anaconda (or in Anaconda Prompt).
After execution, make sure to check for any differences in git-based python code.
These files are included in this folder and are named e.g. "WARNING_GIT_DIFFERENCE_some_package".
If this happens, make sure to change the files in the git-based python packages after installation.
For future use, be sure to commit and push your changes before running any research study.
"""
    _s_path = save_path.joinpath("general_information.txt")
    data = {
        "Time": datetime.now(),
        "Machine": platform.machine(),
        "Version": platform.version(),
        "Platform": platform.platform(),
        "System": platform.system(),
        "Processor": platform.processor(),
    }
    with open(_s_path, "w+") as file:
        file.write(info_header)
        file.write("\nGeneral system information of performed study:\n")
        file.writelines([f"{k}: {v}\n" for k, v in data.items()])
    return _s_path


def get_python_packages(save_path=None):
    _s_path = save_path.joinpath("python_requirements.txt")
    try:
        from pip._internal.utils.misc import get_installed_distributions
    except ImportError:  # pip<10
        from pip import get_installed_distributions
    installed_packages = get_installed_distributions()
    diff_paths = []
    with open(_s_path, "w+") as file:
        for package in installed_packages:
            repo_info = get_git_information(
                path=package.location,
                name=package.key,
                save_path=save_path,
                save_diff=True
            )
            if repo_info is None:
                file.write(f"{package.key}=={package.version}\n")
            else:
                cmt_sha = repo_info["commit"]
                file.write(f"git+{repo_info['url']}.git@{cmt_sha}#egg={package.key}\n")
                diff_paths.extend(repo_info["difference_files"])
    return _s_path, diff_paths


def save_python_reproduction(requirements_path, title=""):
    save_path = pathlib.Path(requirements_path).parent
    _s_path = save_path.joinpath("reproduce_python.bat")
    _v = sys.version_info
    py_version = ".".join([str(_v.major), str(_v.minor), str(_v.micro)])
    env_name = f"py_reproduce_{title}"
    with open(_s_path, "w+") as file:
        file.write(f"conda create -n {env_name} python={py_version} -y\n")
        file.write(f"conda activate {env_name}\n")
        file.write(f"pip install --upgrade pip\n")
        file.write(f"pip install -r {requirements_path}\n")
        file.write(f"conda deactivate\n")
    return _s_path


def get_git_information(path, name=None, save_diff=True, save_path=None, ):
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
    if not save_diff:
        return data

    if save_path is None:
        save_path = pathlib.Path(os.getcwd())
    if name is None:
        name = data["url"].split("/")[0].replace(".git", "")  # Get last part of url
    # Check new files
    if diff_last_cmt:
        _s_path_repo = save_path.joinpath(
            f"WARNING_GIT_DIFFERENCE_{name}_to_local_head.txt"
        )
        data["difference_files"].append(_s_path_repo)
        with open(_s_path_repo, "w+") as diff_file:
            diff_file.write(diff_last_cmt)
    # Check if pushed to remote
    if not repo.git.branch("-r", contains=commit_hex):
        _s_path_repo = save_path.joinpath(
            f"WARNING_GIT_DIFFERENCE_{name}_to_remote_main.txt"
        )
        data["difference_files"].append(_s_path_repo)
        with open(_s_path_repo, "w+") as diff_file:
            diff_file.write(diff_remote_main)
        data["commit"] = remote_main_cmt
    return data


if __name__ == '__main__':
    save_reproduction(
        title="my_study",
        save_path=r"D:\00_temp\reproduction",
    )
