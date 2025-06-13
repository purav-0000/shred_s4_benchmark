import importlib.util
import os
import subprocess
import sys


def install_repo(repo_url, local_dir):
    """Downloads the S4 and SHRED repositories"""
    if not os.path.exists(local_dir):
        print(f"Cloning {repo_url} into {local_dir}...")
        subprocess.run(["git", "clone", repo_url, local_dir], check=True)


def add_to_python_path(file_path):
    """Adds given file path to Python path"""
    if file_path not in sys.path:
        sys.path.insert(0, os.path.abspath(file_path))


"""
def import_module_from_path(alias, file_path, repo_root):
    # Handles imports from S4 and SHRED repositories
    # Add repo root to sys.path if needed
    if repo_root not in sys.path:
        sys.path.insert(0, os.path.abspath(repo_root))

    spec = importlib.util.spec_from_file_location(alias, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
"""