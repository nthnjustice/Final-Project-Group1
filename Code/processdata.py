import os
import shutil
import geopandas as gpd
import shapefile
from PIL import Image, ImageDraw

padus = gpd.read_file('data/shapefiles/PADUS2_0_Shapefiles/PADUS2_0Fee.shp')
target = 'Own_Name'


def init_dir(name):
    path = 'data/' + name
    os.makedirs(path, exist_ok=True)

    if name != 'temp':
        for root, dirs, files in os.walk(path):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))


init_dir('images')
init_dir('temp')
init_dir('train')
init_dir('valid')
init_dir('test')

for i in padus[target].unique():
    init_dir('images/' + i)
    padus[padus[target] == i].to_file('data/temp/temp.shp')

    shpr = shapefile.Reader('data/temp/temp.shp')

    xdist = shpr.bbox[2] - shpr.bbox[0]
    ydist = shpr.bbox[3] - shpr.bbox[1]
    xyratio = xdist / ydist

    max_dim = 500
    width = None
    height = None

    if xyratio >= 1:
        width = max_dim
        height = int(max_dim / xyratio)
    else:
        width = int(max_dim / xyratio)
        height = max_dim

    xratio = width / xdist
    yratio = height / ydist

    count = 0
    for shape in shpr.shapes():
        img = Image.new(mode='RGB', size=(width, height), color='#ffffff')
        draw = ImageDraw.Draw(img)
        pixels = []

        for x, y in shape.points:
            px = int(width - (shpr.bbox[2] - x) * xratio)
            py = int((shpr.bbox[3] - y) * yratio)
            pixels.append((px, py))

        draw.polygon(pixels, outline='#000000')
        img.save('data/images/' + i + '/' + i + '_' + str(count) + '.png')
        count += 1
