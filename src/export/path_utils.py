import os

DEFAULT_RUN_SUBDIR = "nonbatch"


def get_repo_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_data_dir(subdir=""):
    if subdir:
        return os.path.join(get_repo_root(), "data", subdir)
    return os.path.join(get_repo_root(), "data")
