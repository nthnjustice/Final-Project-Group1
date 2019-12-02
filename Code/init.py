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
#   4) generating train/validation/test data splits
########################################################################################################################

class Dataset:
    def __init__(self, shapefile, target, dimension, padding, min_count, valid_size, test_size):
        # initialize list of data-related directories
        self.dirs = ['data', 'images', 'shapefiles', 'test', 'train', 'valid', 'models']
        # initialize file name storing data source names and locations
        self.sources = 'sources.txt'
        # initialize storage for GeoDataFrame object
        self.df = None

        # assign path to shapefile of interest
        self.shapefile = shapefile
        # assign name of feature to use for class labels
        self.target = target
        # assign desired dimension of output images minus padding size
        self.dimension = dimension
        # assign padding size for images
        self.padding = padding
        # assign minimum count of observations for valid class label
        self.min_count = min_count
        # assign validation/test split sizes
        self.valid_size = valid_size
        self.test_size = test_size

    # purpose: call all methods necessary to prepare project
    def run(self):
        self.init_directories()
        self.fetch_data()
        self.set_df()
        self.convert_shp()
        self.split_data()

    # purpose: initialize data-related directories
    def init_directories(self):
        [self.init_directory(i) for i in self.dirs]

    # purpose: fetch and unzip shapefiles from remote sources
    def fetch_data(self):
        # read data source name and location pairs
        lines = [line.rstrip('\n') for line in open(self.sources)]
        i = 0
        while i < len(lines):
            # separate and store data source name and location
            name = lines[i]
            url = lines[i + 1]

            # fetch data from remote location
            res = urllib.request.urlopen(url).read()
            # declare destination path
            path = 'data/shapefiles/' + name

            # save zipped response from remote location
            with open(path + '.zip', 'wb') as file:
                file.write(res)

            # unzip saved data
            zipfile.ZipFile(path + '.zip').extractall(path)
            # increment through name and location pairs
            i += 2

    # purpose: read and assign shapefile of interest
    def set_df(self):
        self.df = gpd.read_file(self.shapefile)

    # purpose: convert spatial polygons to .png images
    def convert_shp(self):
        # assign final dimension of output images
        new_dimension = self.dimension + self.padding

        # loop through class labels of interest
        for label in self.df[self.target].unique():
            # initialize output directory
            self.init_directory('images/' + label)

            # subset observations of class label in focus
            layer = self.df[self.df[self.target] == label]
            # decompose GeoDataFrame into GeoJSON object
            geo = layer.__geo_interface__

            # initialize iterator for output image suffix
            count = 0
            # loop through object features
            for feature in geo['features']:
                # capture properties of feature in focus
                ftype = feature['geometry']['type']
                polygons = feature['geometry']['coordinates']
                bbox = feature['bbox']

                # calculate dimension ratios to be used for sizing output images
                # https://gist.github.com/ianfieldhouse/2284557
                xdist = bbox[2] - bbox[0]
                ydist = bbox[3] - bbox[1]
                xratio = new_dimension / xdist
                yratio = new_dimension / ydist

                # initialize components for creating a new image
                img = Image.new(mode='1', size=(self.dimension, self.dimension), color='#ffffff')
                draw = ImageDraw.Draw(img)
                pixels = []

                if ftype == 'Polygon':
                    # generate image representation of feature polygon in focus
                    # https://gist.github.com/ianfieldhouse/2284557
                    for coords in polygons[0]:
                        px = int(self.dimension - (bbox[2] - coords[0]) * xratio)
                        py = int((bbox[3] - coords[1]) * yratio)
                        pixels.append((px, py))
                    draw.polygon(pixels, outline='#000000', fill='#000000')
                elif ftype == 'MultiPolygon':
                    # generate image representation of feature polygons in focus
                    # https://gist.github.com/ianfieldhouse/2284557
                    for polygon in polygons:
                        for coords in polygon[0]:
                            px = int(self.dimension - (bbox[2] - coords[0]) * xratio)
                            py = int((bbox[3] - coords[1]) * yratio)
                            pixels.append((px, py))
                        draw.polygon(pixels, outline='#000000', fill='#000000')
                        pixels = []

                # apply padding to image of feature polygon(s) in focus
                # https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
                new_img = Image.new(mode='1', size=(new_dimension, new_dimension), color='#ffffff')
                new_img.paste(img, ((new_dimension - self.dimension) // 2, (new_dimension - self.dimension) // 2))

                # save new image
                new_img.save('data/images/' + label + '/' + label + '_' + str(count) + '.png')
                # increment output image suffix
                count += 1

    # purpose: split image data into train/validation/test sets
    def split_data(self):
        root = 'data/images/'
        # initialize output directory
        dirs = os.listdir(root)
        # create list of class labels lacking minimum number of observations
        others = [name for name in dirs if len(os.listdir(root + name)) < self.min_count]

        # initialize storage for input and target label data
        x = []
        y = []

        # loop through output directories for class labels
        for i in dirs:
            images = os.listdir(root + i)
            # use 'OTHR' label if class label in focus lacks minimum number of observations
            label = 'OTHR' if i in others else i

            # populate input and target storage
            for image in images:
                x.append(image)
                y.append(label)

        # convert lists of data to arrays
        x = np.array(x)
        y = np.array(y)

        # generate train/validation/test data splits
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=self.test_size, random_state=0)
        xtrain, xvalid, ytrain, yvalid = train_test_split(xtrain, ytrain, test_size=self.valid_size, random_state=0)

        # rebuild list of target class label directories
        dirs = [i for i in dirs if i not in others]
        if len(others) > 0:
            dirs.append('OTHR')

        # move image splits to their appropriate destination
        self.move_images('train/', dirs, xtrain, ytrain)
        self.move_images('test/', dirs, xtest, ytest)
        self.move_images('valid/', dirs, xvalid, yvalid)

    # purpose: move images from one directory to another
    # inputs:
    #   1) root path of output directory
    #   2) list of sub-level output directories
    #   3) array of image file names to be moved
    #   4) array of sub-level directory names to be used as corresponding destinations
    def move_images(self, path, dirs, x, y):
        # initialize output directories
        [self.init_directory(path + i) for i in dirs]

        # loop through arrays and move images to the appropriate location
        for i in range(len(x)):
            source = 'data/images/' + x[i].split('_')[0] + '/' + x[i]
            destination = 'data/' + path + y[i] + '/' + x[i]
            shutil.copyfile(source, destination)

    # purpose: initialize new directory
    # input: name of directory to be created
    @staticmethod
    def init_directory(name):
        # append 'data/' prefix if appropriate
        path = name if name == 'data' else 'data/' + name
        # create new directory
        os.makedirs(path, exist_ok=True)

        # delete contents of directory if it already exists
        # https://stackoverflow.com/questions/185936/how-to-delete-the-contents-of-a-folder-in-python
        for root, dirs, files in os.walk(path):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))
