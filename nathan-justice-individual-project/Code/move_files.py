# helper function to move files from one directory to another (handles duplicate names without overwriting)

import os
import shutil

from utils.init_dir import init_dir


def move_files(in_root, out_root, dirs, names, labels):
    [init_dir(out_root + '/' + i) for i in dirs]

    for i in range(len(names)):
        source = in_root + '/' + labels[i] + '/' + names[i]
        destination = out_root + '/' + labels[i] + '/' + names[i]

        # https://stackoverflow.com/questions/33282647/python-shutil-copy-if-i-have-a-duplicate-file-will-it-copy-to-new-location
        if not os.path.exists(destination):
            shutil.copyfile(source, destination)
        else:
            base, extension = os.path.splitext(names[i])
            count = 1
            while os.path.exists(destination):
                destination += base + '_' + str(count) + extension
                count += 1
            shutil.copyfile(source, destination)
