import urllib
import zipfile


def run():
    lines = [line.rstrip('\n') for line in open('sources.txt')]

    i = 0
    while i < len(lines):
        name = lines[i]
        url = lines[i + 1]

        res = urllib.request.urlopen(url).read()
        path = 'data/shapefiles/' + name

        with open(path + '.zip', 'wb') as file:
            file.write(res)

        zipfile.ZipFile(path + '.zip').extractall(path)
        i += 2
