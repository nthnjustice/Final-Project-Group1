########################################################################################################################
# LOAD DEPENDENCIES
########################################################################################################################

import os
import shutil
import urllib
import zipfile
import geopandas as gpd
from PIL import Image, ImageDraw
import numpy as np
from sklearn.model_selection import train_test_split


########################################################################################################################
# DATASET OBJECT
# description: prepares project by:
#   1) initializing folder architecture
#   2) loading remote data
#   3) converting shapefile polygons to images
#   4) generating train/test/validation data splits
########################################################################################################################

class Dataset:
    def __init__(self, shapefile, target, dimension, padding, min_count, test_size, valid_size):
        self.dirs = ['data', 'images', 'shapefiles', 'test', 'train', 'valid', 'models']
        self.sources = 'sources.txt'
        self.df = None

        self.shapefile = shapefile
        self.target = target
        self.dimension = dimension
        self.padding = padding
        self.min_count = min_count
        self.test_size = test_size
        self.valid_size = valid_size

    def run(self):
        self.init_directories()
        self.fetch_data()
        self.set_df()
        self.shp2png()
        self.split_data()

    def init_directories(self):
        [self.init_directory(i) for i in self.dirs]

    def fetch_data(self):
        lines = [line.rstrip('\n') for line in open(self.sources)]
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

    def set_df(self):
        self.df = gpd.read_file(self.shapefile)

    def shp2png(self):
        new_dimension = self.dimension + self.padding

        for label in self.df[self.target].unique():
            self.init_directory('images/' + label)

            layer = self.df[self.df[self.target] == label]
            geo = layer.__geo_interface__

            count = 0
            for feature in geo['features']:
                ftype = feature['geometry']['type']
                polygons = feature['geometry']['coordinates']
                bbox = feature['bbox']

                xdist = bbox[2] - bbox[0]
                ydist = bbox[3] - bbox[1]
                xratio = new_dimension / xdist
                yratio = new_dimension / ydist

                img = Image.new(mode='1', size=(self.dimension, self.dimension), color='#ffffff')
                draw = ImageDraw.Draw(img)
                pixels = []

                if ftype == 'Polygon':
                    for coords in polygons[0]:
                        px = int(self.dimension - (bbox[2] - coords[0]) * xratio)
                        py = int((bbox[3] - coords[1]) * yratio)
                        pixels.append((px, py))
                    draw.polygon(pixels, outline='#000000', fill='#000000')
                elif ftype == 'MultiPolygon':
                    for polygon in polygons:
                        for coords in polygon[0]:
                            px = int(self.dimension - (bbox[2] - coords[0]) * xratio)
                            py = int((bbox[3] - coords[1]) * yratio)
                            pixels.append((px, py))
                        draw.polygon(pixels, outline='#000000', fill='#000000')
                        pixels = []

                new_img = Image.new(mode='1', size=(new_dimension, new_dimension), color='#ffffff')
                new_img.paste(img, ((new_dimension - self.dimension) // 2, (new_dimension - self.dimension) // 2))

                new_img.save('data/images/' + label + '/' + label + '_' + str(count) + '.png')
                count += 1

    def split_data(self):
        root = 'data/images/'
        dirs = os.listdir(root)
        others = [name for name in dirs if len(os.listdir(root + name)) < self.min_count]

        x = []
        y = []

        for i in dirs:
            images = os.listdir(root + i)
            label = 'OTHR' if i in others else i

            for image in images:
                x.append(image)
                y.append(label)

        x = np.array(x)
        y = np.array(y)

        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=self.test_size, random_state=0)
        xtrain, xvalid, ytrain, yvalid = train_test_split(xtrain, ytrain, test_size=self.valid_size, random_state=0)

        dirs = [i for i in dirs if i not in others]
        dirs.append('OTHR')

        self.move_images('train/', dirs, xtrain, ytrain)
        self.move_images('test/', dirs, xtest, ytest)
        self.move_images('valid/', dirs, xvalid, yvalid)

    def move_images(self, path, dirs, x, y):
        [self.init_directory(path + i) for i in dirs]

        for i in range(len(x)):
            source = 'data/images/' + x[i].split('_')[0] + '/' + x[i]
            destination = 'data/' + path + y[i] + '/' + x[i]
            shutil.copyfile(source, destination)

    @staticmethod
    def init_directory(name):
        path = name if name == 'data' else 'data/' + name
        os.makedirs(path, exist_ok=True)

        for root, dirs, files in os.walk(path):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))
