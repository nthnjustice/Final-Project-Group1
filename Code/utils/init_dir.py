# helper function to create new directory or delete contents if it already exists

import os
import shutil


def init_dir(path):
    os.makedirs(path, exist_ok=True)

    # https://stackoverflow.com/questions/185936/how-to-delete-the-contents-of-a-folder-in-python
    for root, dirs, files in os.walk(path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
