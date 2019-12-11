# converts geospatial data into png images

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

    dirs = [d for d in os.listdir(data_path) if not d.endswith('.zip')]

    if dirs[0] == 'buildings':
        d_path = data_path + '/' + [d for d in os.listdir(data_path) if not d.endswith('.zip')][0]

        # loop through target labels
        for f in os.listdir(d_path):
            label = f.split('.')[0]
            label = label.split('-')[1]
            label = get_buildfunc(label)
            init_dir(out_root + '/' + label)

            # load observations with label in focus
            df = pd.read_csv(d_path + '/' + f)
            shapes = [wkt.loads(geom) for geom in df['geometrie']]

            # loop through observations
            count = 0
            for shape in shapes:
                geo = shape.__geo_interface__

                if geo['type'] == 'Polygon':
                    polygons = geo['coordinates'][0]
                else:
                    polygons = geo['coordinates'][0][0]

                # store geometry coordiante values
                x_coords = []
                y_coords = []

                for polygon in polygons:
                    x_coords.append(polygon[0])
                    y_coords.append(polygon[1])

                # https://gist.github.com/ianfieldhouse/2284557
                # store values defining the geospatial bounding box
                min_x = shape.bounds[0]
                max_x = shape.bounds[2]
                min_y = shape.bounds[1]
                max_y = shape.bounds[3]

                # calculate the distance between the minimum and maximum coordinate observation for each axis
                x_dist = max_x - min_x
                y_dist = max_y - min_y

                # calculate the ratio of the desired output dimension and the min/max distance for each axis
                x_ratio = img_dim / x_dist
                y_ratio = img_dim / y_dist

                img = Image.new(mode=img_mode, size=(img_dim, img_dim), color='#ffffff')
                draw = ImageDraw.Draw(img)
                pixels = []

                # loop through coordinate pairs defining the geospatial geometry of the feature/polygon in focus
                for coords in polygons:
                    # scale coordinates to values that fit on a 100x100 image/grid
                    px = int(img_dim - (max_x - coords[0]) * x_ratio)
                    py = int((max_y - coords[1]) * y_ratio)
                    pixels.append((px, py))
                draw.polygon(pixels, outline='#000000', fill=img_fill)

                img.save(out_root + '/' + label + '/' + label + '_' + str(count) + '.png')
                count += 1
    elif dirs[0] == 'neighborhoods':
        df = pd.read_csv(data_path + '/neighborhoods/neighborhoods.csv')
        df = df[df['aantal_inwoners'] >= 0]
        median = df['aantal_inwoners'].median()
        df['target'] = df['aantal_inwoners'].apply(lambda x: 'over' if x >= median else 'under')

        # see comments above for annotations
        for label in df['target'].unique():
            init_dir(out_root + '/' + label)

            layer = df[df['target'] == label]
            shapes = [wkt.loads(geom) for geom in layer['geom']]

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

    # else:
        # for label in dirs:
        #     init_dir(out_root + '/' + label)
        #
        #     df = gpd.read_file(data_path + '/' + label + '/' + label + '.shp')
        #     geo = df.__geo_interface__
        #
        #     count = 0
        #     for feature in geo['features']:
        #         ftype = feature['geometry']['type']
        #         polygons = feature['geometry']['coordinates']
        #         bbox = feature['bbox']
        #
        #         # https://gist.github.com/ianfieldhouse/2284557
        #         min_x = bbox[0]
        #         max_x = bbox[2]
        #         min_y = bbox[1]
        #         max_y = bbox[3]
        #
        #         x_dist = max_x - min_x
        #         y_dist = max_y - min_y
        #
        #         x_ratio = img_dim / x_dist
        #         y_ratio = img_dim / y_dist
        #
        #         img = Image.new(mode=img_mode, size=(img_dim, img_dim), color='#ffffff')
        #         draw = ImageDraw.Draw(img)
        #         pixels = []
        #
        #         if ftype == 'Polygon':
        #             for coords in polygons[0]:
        #                 px = int(img_dim - (max_x - coords[0]) * x_ratio)
        #                 py = int((max_y - coords[1]) * y_ratio)
        #                 pixels.append((px, py))
        #             draw.polygon(pixels, outline='#000000', fill=img_fill)
        #         elif ftype == 'MultiPolygon':
        #             for polygon in polygons:
        #                 for coords in polygon[0]:
        #                     px = int(img_dim - (max_x - coords[0]) * x_ratio)
        #                     py = int((max_y - coords[1]) * y_ratio)
        #                     pixels.append((px, py))
        #                 draw.polygon(pixels, outline='#000000', fill=img_fill)
        #
        #         img.save(out_root + '/' + label + '/' + label + '_' + str(count) + '.png')
        #         count += 1
