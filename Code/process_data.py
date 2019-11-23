from init_directories import init_dir
import geopandas as gpd
from PIL import Image, ImageDraw


def run():
    padus = gpd.read_file('data/shapefiles/PADUS2_0_Shapefiles/PADUS2_0Fee.shp')
    target = 'Own_Type'
    dim = 250

    for label in padus[target].unique():
        imgpath = 'images/' + label
        init_dir(imgpath)

        layer = padus[padus[target] == label]
        geo = layer.__geo_interface__

        count = 0
        for feature in geo['features']:
            ftype = feature['geometry']['type']
            polygons = feature['geometry']['coordinates']
            bbox = feature['bbox']

            xdist = bbox[2] - bbox[0]
            ydist = bbox[3] - bbox[1]
            xratio = dim / xdist
            yratio = dim / ydist

            img = Image.new(mode='RGB', size=(dim, dim), color='#ffffff')
            draw = ImageDraw.Draw(img)
            pixels = []

            if ftype == 'Polygon':
                for coords in polygons[0]:
                    px = int(dim - (bbox[2] - coords[0]) * xratio)
                    py = int((bbox[3] - coords[1]) * yratio)
                    pixels.append((px, py))
                draw.polygon(pixels, outline='#000000')
            elif ftype == 'MultiPolygon':
                for polygon in polygons:
                    for coords in polygon[0]:
                        px = int(dim - (bbox[2] - coords[0]) * xratio)
                        py = int((bbox[3] - coords[1]) * yratio)
                        pixels.append((px, py))
                    draw.polygon(pixels, outline='#000000')
                    pixels = []

            img.save('data/' + imgpath + '/' + label + '_' + str(count) + '.png')
            count += 1
