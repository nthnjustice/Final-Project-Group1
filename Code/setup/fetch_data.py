import os
from urllib import request
import zipfile


def fetch_data(sources, out_path):
    if not os.path.isfile(sources):
        raise FileNotFoundError("Can't find {}".format(sources))
    elif not os.path.exists(out_path):
        raise Exception("{} doesn't exist".format(out_path))

    lines = [line.rstrip('\n') for line in open(sources)]
    i = 0

    while i < len(lines):
        name = lines[i]
        url = lines[i + 1]

        res = request.urlopen(url).read()
        path = out_path + '/' + name

        with open(path + '.zip', 'wb') as file:
            file.write(res)
        zipfile.ZipFile(path + '.zip').extractall(path)

        i += 2
