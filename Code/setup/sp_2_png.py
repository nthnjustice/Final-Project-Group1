import os
import pandas as pd
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

    for d in [d for d in os.listdir(data_path) if not d.endswith('.zip')]:
        d_path = data_path + '/' + d

        if d == 'buildings':
            for f in os.listdir(d_path):
                label = f.split('.')[0]
                label = label.split('-')[1]
                label = get_buildfunc(label)
                init_dir(out_root + '/' + label)

                df = pd.read_csv(d_path + '/' + f)
                shapes = [wkt.loads(geom) for geom in df.geometrie.values]

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

                    min_x = min(x_coords)
                    max_x = max(x_coords)
                    min_y = min(y_coords)
                    max_y = max(y_coords)

                    x_ratio = dim / (max_x - min_x)
                    y_ratio = dim / (max_y - min_y)

                    img = Image.new(mode=img_mode, size=(img_dim, img_dim), color='#ffffff')
                    draw = ImageDraw.Draw(img)
                    pixels = []

                    for coords in polygons:
                        px = int(img_dim - (max_x - coords[0]) * x_ratio)
                        py = int((max_y - coords[1]) * y_ratio)
                        pixels.append((px, py))
                    draw.polygon(pixels, outline='#000000', fill=img_fill)

                    # https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
                    new_img = Image.new(mode=img_mode, size=(dim, dim), color='#ffffff')
                    inner_x = (dim - img_dim) // 2
                    inner_y = (dim - img_dim) // 2
                    new_img.paste(img, (inner_x, inner_y))

                    new_img.save(out_root + '/' + label + '/' + label + '_' + str(count) + '.png')
                    count += 1
