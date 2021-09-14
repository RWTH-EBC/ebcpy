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
import atexit
from datetime import datetime
from git import Repo, InvalidGitRepositoryError, RemoteReference
from ebcpy import DymolaAPI, FMU_API

logger = logging.getLogger(__name__)


def register(title=None, save_path=None, sim_api=None):
    if save_path is None:
        save_path = os.getcwd()
    if title is None:
        title = pathlib.Path(__file__).name.replace(".py", "")
    atexit.register(
        save_reproduction,
        save_path=save_path,
        title=title,
        sim_api=sim_api
    )


def save_reproduction(title, save_path, sim_api=None):
    save_path = pathlib.Path(save_path)
    os.makedirs(save_path, exist_ok=True)
    files_to_save = []
    # Start with the file currently running:
    file_running = pathlib.Path(__file__).absolute()
    file_running_save = save_path.joinpath(file_running.name)
    copy_file(src=file_running, dst=file_running_save)
    files_to_save.append(file_running_save)
    # General info
    files_to_save.append(
        save_general_information(save_path=save_path)
    )
    # Python-Reproduction:
    requirements_path, diff_files = get_python_packages(save_path=save_path)
    files_to_save.extend(diff_files)
    files_to_save.append(requirements_path)
    files_to_save.append(
        save_python_reproduction(
            requirements_path=requirements_path,
            title=title
        )
    )
    # Simulation reproduction:
    if isinstance(sim_api, DymolaAPI):
        m_name = sim_api.model_name
        f_name = save_path.joinpath(f"{m_name}_total.mo")
        # Total model
        res = sim_api.dymola.saveTotalModel(
            fileName=f_name,
            modelName=m_name
        )
        if res:
            files_to_save.append(f_name)
        else:
            logger.error("Could not save total model: %s",
                         sim_api.dymola.getLastErrorLog())
        # FMU
        res = sim_api.dymola.translateModelFMU(
            modelToOpen=sim_api.model_name,
            storeResult=False,
            modelName='',
            fmiVersion='1',
            fmiType='all',
            includeSource=False,
            includeImage=0
        )
        if res:
            files_to_save.append(res)
        else:
            logger.error("Could not export fmu: %s",
                         sim_api.dymola.getLastErrorLog())
    elif isinstance(sim_api, FMU_API):
        # Directly copy and save the FMU in use:
        fmu_name_save = save_path.joinpath(
            pathlib.Path(sim_api.model_name).name
        )
        copy_file(src=fmu_name_save, dst=sim_api.model_name)
        files_to_save.append(fmu_name_save)

    zip_file = save_to_zip(
        files=files_to_save,
        title=title,
        save_path=save_path
    )
    # Remove created files:
    for file in files_to_save:
        os.remove(file)
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
            if is_git_repo(package.location):
                repo = get_git_information(package.location)
                if not repo['clean']:
                    _s_path_repo = save_path.joinpath(
                        f"WARNING_GIT_DIFFERENCE_{package.key}_to_local_head.txt"
                    )
                    diff_paths.append(_s_path_repo)
                    with open(_s_path_repo, "w+") as diff_file:
                        diff_file.write(repo['diff'])
                if not repo['pushed']:
                    _s_path_repo = save_path.joinpath(
                        f"WARNING_GIT_DIFFERENCE_{package.key}_to_remote_main.txt"
                    )
                    diff_paths.append(_s_path_repo)
                    with open(_s_path_repo, "w+") as diff_file:
                        diff_file.write(repo['diff_remote_main'])
                    cmt_sha = repo['remote_main_commit']
                else:
                    cmt_sha = repo['commit']
                file.write(f"git+{repo['url']}.git@{cmt_sha}#egg={package.key}\n")
            else:
                file.write(f"{package.key}=={package.version}\n")
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


def get_git_information(path):
    repo = Repo(path)
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
        "clean": diff_last_cmt == '',
        "diff": diff_last_cmt,
        "pushed": repo.git.branch("-r", contains=commit_hex) != '',
        "diff_remote_main": diff_remote_main,
        "remote_main_commit": remote_main_cmt
    }
    return data
    # ...


def is_git_repo(path):
    """Return true if given path is a git repo"""
    try:
        Repo(path)
        return True
    except InvalidGitRepositoryError:
        return False


if __name__ == '__main__':
    save_reproduction(
        title="my_study",
        save_path=r"D:\00_temp\reproduction",
    )
