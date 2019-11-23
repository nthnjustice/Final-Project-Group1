import os
import shutil


def run():
    init_dir('data')
    init_dir('shapefiles')
    init_dir('images')
    init_dir('train')
    init_dir('valid')
    init_dir('test')


def init_dir(name):
    path = 'data' if name == 'data' else 'data/' + name
    os.makedirs(path, exist_ok=True)

    for root, dirs, files in os.walk(path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
