import os
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely import wkt
from PIL import Image, ImageDraw

from utils.get_buildfunc import get_buildfunc
from utils.init_dir import init_dir


def sp_2_png(img_dim, img_mode, img_fill):
    data_path = 'data/spatial'
    out_root = 'data/images'

    if not os.path.exists(data_path):
        raise Exception("{} doesn't exist".format(data_path))
    elif not os.path.exists(out_root):
        raise Exception("{} doesn't exist".format(out_root))

    d = [d for d in os.listdir(data_path) if not d.endswith('.zip')][0]
    d_path = data_path + '/' + d

    if d == 'buildings':
        for f in os.listdir(d_path):
            label = f.split('.')[0]
            label = label.split('-')[1]
            label = get_buildfunc(label)
            init_dir(out_root + '/' + label)

            df = pd.read_csv(d_path + '/' + f)
            shapes = [wkt.loads(geom) for geom in df['geometrie']]

            count = 0
            for shape in shapes:
                geo = shape.__geo_interface__

                if geo['type'] == 'Polygon':
                    polygons = geo['coordinates'][0]
                else:
                    polygons = geo['coordinates'][0][0]

                x_coords = []
                y_coords = []

                for polygon in polygons:
                    x_coords.append(polygon[0])
                    y_coords.append(polygon[1])

                min_x = shape.bounds[0]
                max_x = shape.bounds[2]
                min_y = shape.bounds[1]
                max_y = shape.bounds[3]

                x_dist = max_x - min_x
                y_dist = max_y - min_y

                x_ratio = img_dim / x_dist
                y_ratio = img_dim / y_dist

                img = Image.new(mode=img_mode, size=(img_dim, img_dim), color='#ffffff')
                draw = ImageDraw.Draw(img)
                pixels = []

                for coords in polygons:
                    px = int(img_dim - (max_x - coords[0]) * x_ratio)
                    py = int((max_y - coords[1]) * y_ratio)
                    pixels.append((px, py))
                draw.polygon(pixels, outline='#000000', fill=img_fill)

                img.save(out_root + '/' + label + '/' + label + '_' + str(count) + '.png')
                count += 1
