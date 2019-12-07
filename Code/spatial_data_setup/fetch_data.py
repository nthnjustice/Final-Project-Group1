import os
from urllib import request
import zipfile


def fetch_data(name, url, out_path):
    if not os.path.exists(out_path):
        raise Exception("{} doesn't exist".format(out_path))

    res = request.urlopen(url).read()
    path = out_path + '/' + name

    with open(path + '.zip', 'wb') as file:
        file.write(res)
    zipfile.ZipFile(path + '.zip').extractall(path)