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
        title: str = None,
        path: Union[pathlib.Path, str] = None,
        log_message: str = None,
        files: List[Union[ReproductionFile, CopyFile]] = None,
        file: Union[pathlib.Path, str] = None,
        search_on_pypi: bool = False
):
    """
    Function to save a reproduction archive which contains
    files to reproduce any simulation/software based study.

    :param str title:
        Title of the study
    :param pathlib.Path path:
        Where to store the .zip file. If not given, os.getcwd() is used.
    :param str log_message:
         Specific message for this run of the study. If given,
         you are not asked at the end of your script to give the
         log_message.
    :param list files:
        List of files to save along the standard ones.
        Examples would be plots, tables etc.
    :param pathlib.Path file:
        The file which is used to run.
        Default is __file__ of __main__ module
    :param bool search_on_pypi:
        If True, all python packages which are
        not a git-repo are checked for availability on pypi
        Default is False. Does not work if no internet connection
        is available.
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
            zip_folder_path="study_repository"
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
    py_requirements_content, diff_files, pip_version = _get_python_package_information(
        search_on_pypi=search_on_pypi
    )
    files.append(ReproductionFile(
        filename=_py_requirements_name,
        content=py_requirements_content,
    ))
    files.extend(diff_files)

    py_repro = _get_python_reproduction(
        title=title,
        pip_version=pip_version
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
                else:
                    logger.error("Given file '%s' is a string but "
                                 "not an existing file. Skipping...", file)
            elif isinstance(file, ReproductionFile):
                zip_file.writestr(file.filename, file.content)
            elif isinstance(file, CopyFile):
                zip_file.write(file.sourcepath, file.filename)
                if file.remove:
                    try:
                        os.remove(file.sourcepath)
                    except PermissionError:
                        logger.error(f"Could not remove {file.sourcepath}")
            else:
                raise TypeError(
                    f"Given file '{file}' has no "
                    f"valid type. Type is '{type(file)}'")
    return zip_file_name


def get_git_information(
        path: pathlib.Path,
        name: str = None,
        zip_folder_path: str = None
):
    """
    Function to get the git information for a given path.

    :param pathlib.Path path:
        Path to possible git repo
    :param str name:
        Name of the repo.
        If not given, the name in the URL will be used.
    :param str zip_folder_path:
        If given, the PATH of the difference_files for the .zip
        will be zip_folder_path plus WARNING_GIT_DIFFERENCE...

    Returns:
        If the path is not a git repository, this function returns None.
        Else, a dictionary with the keys 'url', 'commit' and 'difference_files'.
    """
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
    if zip_folder_path is None:
        zip_folder_path = ""
    else:
        zip_folder_path += "/"
    # Check new files
    if diff_last_cmt:
        data["difference_files"].append(ReproductionFile(
            filename=f"{zip_folder_path}WARNING_GIT_DIFFERENCE_{name}_to_local_head.txt",
            content=diff_last_cmt,
        ))
    # Check if pushed to remote
    if not repo.git.branch("-r", contains=commit_hex):
        data["difference_files"].append(ReproductionFile(
            filename=f"{zip_folder_path}WARNING_GIT_DIFFERENCE_{name}_to_remote_main.txt",
            content=diff_remote_main,
        ))
        data["commit"] = remote_main_cmt
    return data


def creat_copy_files_from_dir(foldername: str,
                              sourcepath: pathlib.Path,
                              remove: bool = False):
    """
    Creates a list with CopyFiles for each file in a directory
    where which will be saved in the zip under the foldername
    with all subdirectories.

    :param str foldername:
        Name of the folder in the zip. Can be a relative path.
    :param pathlib.Path sourcepath:
        Path on the current machine where the directory to copy
        is located
    :param bool remove:
        Default is False. If True, the files in the directory
        will be moved instead of just copied.

    :return list:
        Returns a list with CopyFiles for each file in the directory source path.
    """
    files = []
    for dirpath, dirnames, filenames in os.walk(sourcepath):
        for file in filenames:
            filename = foldername + dirpath.__str__().split(sourcepath.name)[-1] + '/' + file
            files.append(CopyFile(
                sourcepath=os.path.join(dirpath, file),
                filename=filename,
                remove=remove
            ))
    return files


def _get_general_information(title: str, log_message: str, current_time: str):
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
    import pkg_resources
    installed_packages = [pack for pack in pkg_resources.working_set]
    diff_paths = []
    requirement_txt_content = []
    pip_version = ""
    for package in installed_packages:
        repo_info = get_git_information(
            path=package.location,
            name=package.key,
            zip_folder_path="python"
        )
        if repo_info is None:
            # Check if in python path:
            if package.key == "pip":  # exclude pip in requirements and give info to _get_python_reproduction
                pip_version = f"=={package.version}"
            else:
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
    return "\n".join(requirement_txt_content), diff_paths, pip_version


def _get_python_reproduction(title: str, pip_version: str):
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
        f"python -m pip install pip{pip_version}",
        f"pip install -r requirements.txt",
    ]
    return "\n".join(py_reproduce_content)


if __name__ == '__main__':
    save_reproduction_archive(
        title="my_study",
        path=r"D:\00_temp\reproduction",
    )
