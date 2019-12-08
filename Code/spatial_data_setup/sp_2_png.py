import os
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely import wkt
from PIL import Image, ImageDraw

from utils.get_buildfunc import get_buildfunc
from utils.init_dir import init_dir


def sp_2_png(img_dim, img_pad, img_mode, img_fill, target):
    data_path = 'data/spatial'
    out_root = 'data/images'

    if not os.path.exists(data_path):
        raise Exception("{} doesn't exist".format(data_path))
    elif not os.path.exists(out_root):
        raise Exception("{} doesn't exist".format(out_root))

    dim = img_dim + img_pad

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
                xy_ratio = x_dist / y_dist

                if xy_ratio >= 1:
                    iwidth = dim
                    iheight = int(dim / xy_ratio)
                else:
                    iwidth = int(dim / xy_ratio)
                    iheight = dim

                x_ratio = 100 / x_dist
                y_ratio = 100 / y_dist

                img = Image.new(mode=img_mode, size=(dim, dim), color='#ffffff')
                draw = ImageDraw.Draw(img)
                pixels = []

                for coords in polygons:
                    px = int(dim - (max_x - coords[0]) * x_ratio)
                    py = int((max_y - coords[1]) * y_ratio)
                    pixels.append((px, py))
                draw.polygon(pixels, outline='#000000', fill=img_fill)

                # https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
                # new_img = Image.new(mode=img_mode, size=(dim, dim), color='#ffffff')
                # inner_x = (dim - img_dim) // 2
                # inner_y = (dim - img_dim) // 2
                # new_img.paste(img, (inner_x, inner_y))

                #new_
                img.save(out_root + '/' + label + '/' + label + '_' + str(count) + '.png')
                count += 1
    # elif d == 'archaeology':
    #     df = pd.read_csv(d_path + '/archaeology.csv')
    #
    #     for label in df['Aardspoor'].unique():
    #         if type(label) is not str:
    #             if np.isnan(label) == True:
    #                 continue
    #
    #         init_dir(out_root + '/' + label)
    #         layer = df[df['Aardspoor'] == label]
    #         layer.reset_index(inplace=True)
    #         shapes = []
    #
    #         for i in range(len(layer)):
    #             if label == 'XXX' and i == 398:
    #                 continue
    #
    #             poly = layer.loc[i, 'WKT'].split(',')
    #
    #             if len(poly) > 4:
    #                 shapes.append(wkt.loads(layer.loc[i, 'WKT']))
    #
    #         count = 0
    #         for shape in shapes:
    #             geo = shape.__geo_interface__
    #
    #             if geo['type'] == 'Polygon':
    #                 polygons = geo['coordinates'][0]
    #             else:
    #                 polygons = geo['coordinates'][0][0]
    #
    #             x_coords = []
    #             y_coords = []
    #
    #             for polygon in polygons:
    #                 x_coords.append(polygon[0])
    #                 y_coords.append(polygon[1])
    #
    #             min_x = min(x_coords)
    #             max_x = max(x_coords)
    #             min_y = min(y_coords)
    #             max_y = max(y_coords)
    #
    #             if min_x == max_x or min_y == max_y:
    #                 continue
    #
    #             x_ratio = dim / (max_x - min_x)
    #             y_ratio = dim / (max_y - min_y)
    #
    #             img = Image.new(mode=img_mode, size=(img_dim, img_dim), color='#ffffff')
    #             draw = ImageDraw.Draw(img)
    #             pixels = []
    #
    #             for coords in polygons:
    #                 px = int(img_dim - (max_x - coords[0]) * x_ratio)
    #                 py = int((max_y - coords[1]) * y_ratio)
    #                 pixels.append((px, py))
    #             draw.polygon(pixels, outline='#000000', fill=img_fill)
    #
    #             # https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
    #             new_img = Image.new(mode=img_mode, size=(dim, dim), color='#ffffff')
    #             inner_x = (dim - img_dim) // 2
    #             inner_y = (dim - img_dim) // 2
    #             new_img.paste(img, (inner_x, inner_y))
    #
    #             new_img.save(out_root + '/' + label + '/' + label + '_' + str(count) + '.png')
    #             count += 1
    # else:
    #     df = None
    #
    #     if d == 'PADUS2_0_Shapefiles':
    #         df = gpd.read_file(d_path + '/' + 'PADUS2_0Fee.shp')
    #     elif d == 'MTPublicLands_SHP':
    #         df = gpd.read_file(d_path + '/' + d + '.shp')
    #     elif d == 'Species_CONUS_Range_2001v1':
    #         df = gpd.read_file(d_path + '/' + d + '.shp')
    #     elif d == 'IUCN':
    #         df = gpd.read_file(d_path + '/' + d + '.shp')
    #
    #     if df is None:
    #         raise Exception("Unable to load shapefile in {}".format(d))
    #
    #     for label in df[target].unique():
    #         init_dir(out_root + '/' + label)
    #
    #         layer = df[df[target] == label]
    #         geo = layer.__geo_interface__
    #
    #         count = 0
    #         for feature in geo['features']:
    #             ftype = feature['geometry']['type']
    #             polygons = feature['geometry']['coordinates']
    #             bbox = feature['bbox']
    #
    #             # https://gist.github.com/ianfieldhouse/2284557
    #             xdist = bbox[2] - bbox[0]
    #             ydist = bbox[3] - bbox[1]
    #             x_ratio = dim / xdist
    #             y_ratio = dim / ydist
    #
    #             img = Image.new(mode=img_mode, size=(img_dim, img_dim), color='#ffffff')
    #             draw = ImageDraw.Draw(img)
    #             pixels = []
    #
    #             # generate scaled image representation of feature polygon(s) in focus
    #             # https://gist.github.com/ianfieldhouse/2284557
    #             if ftype == 'Polygon':
    #                 for coords in polygons[0]:
    #                     px = int(img_dim - (bbox[2] - coords[0]) * x_ratio)
    #                     py = int((bbox[3] - coords[1]) * y_ratio)
    #                     pixels.append((px, py))
    #                 draw.polygon(pixels, outline='#000000', fill=img_fill)
    #             elif ftype == 'MultiPolygon':
    #                 for polygon in polygons:
    #                     for coords in polygon[0]:
    #                         px = int(img_dim - (bbox[2] - coords[0]) * x_ratio)
    #                         py = int((bbox[3] - coords[1]) * y_ratio)
    #                         pixels.append((px, py))
    #                     draw.polygon(pixels, outline='#000000', fill=img_fill)
    #                     pixels = []
    #
    #             # https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
    #             new_img = Image.new(mode=img_mode, size=(dim, dim), color='#ffffff')
    #             inner_x = (dim - img_dim) // 2
    #             inner_y = (dim - img_dim) // 2
    #             new_img.paste(img, (inner_x, inner_y))
    #
    #             new_img.save(out_root + '/' + label + '/' + label + '_' + str(count) + '.png')
    #             count += 1