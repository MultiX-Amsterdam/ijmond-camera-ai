"""
Helper functions
"""
import json
import os


def load_json(fpath):
    with open(fpath, "r") as f:
        return json.load(f)


def is_file_here(file_path):
    return os.path.isfile(file_path)