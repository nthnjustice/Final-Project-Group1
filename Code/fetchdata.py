import os
import urllib
import zipfile

root = 'data/shapefiles'
os.makedirs(root, exist_ok=True)

lines = [line.rstrip('\n') for line in open('sources.txt')]

i = 0
while i < len(lines):
    name = lines[i]
    url = lines[i + 1]

    res = urllib.request.urlopen(url).read()
    path = root + '/' + name

    with open(path + '.zip', 'wb') as file:
        file.write(res)

    zipfile.ZipFile(path + '.zip').extractall(path)
    i += 2
