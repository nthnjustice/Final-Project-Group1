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
from imblearn.over_sampling import RandomOverSampler


########################################################################################################################
# DATASET OBJECT
# description: prepares the project via:
#   1) initialize folder architecture
#   2) load remote data
#   3) convert shapefile polygons to images
#   4) generate train/validation/test data splits
#   5) preprocess train split
########################################################################################################################

class Dataset:
    def __init__(self, params):
        # assign parameter values
        self.params = params
        # initialize storage for GeoDataFrame object
        self.df = None

    # purpose: call all methods necessary to prepare the project
    def run(self):
        self.init_directories()
        self.fetch_data()
        self.set_df()
        self.convert_shp()
        self.split_data()

    # purpose: initialize data-related directories
    def init_directories(self):
        [self.init_directory(i) for i in self.params['data_dirs']]

    # purpose: fetch and unzip data from remote sources
    def fetch_data(self):
        # read data-source name and location pairs
        lines = [line.rstrip('\n') for line in open(self.params['sources'])]
        # initialize line iterator
        i = 0
        # loop through name and location pairs
        while i < len(lines):
            # separate and store name and location in focus
            name = lines[i]
            url = lines[i + 1]

            # GET data from remote location
            res = urllib.request.urlopen(url).read()
            # build path to destination folder
            path = 'data/shapefiles/' + name

            # save and unzip response from remote location
            with open(path + '.zip', 'wb') as file:
                file.write(res)
            zipfile.ZipFile(path + '.zip').extractall(path)

            # increment to next name and location pair
            i += 2

    # purpose: load shapefile of interest
    def set_df(self):
        self.df = gpd.read_file(self.params['shapefile'])

    # purpose: convert shapefile polygons to .png images
    def convert_shp(self):
        # calculate final dimension of output images
        new_dimension = self.params['dimension'] + self.params['padding']

        # loop through class labels
        for label in self.df[self.params['target']].unique():
            # initialize output directory for label in focus
            self.init_directory('images/' + label)

            # subset observations of class label in focus
            layer = self.df[self.df[self.params['target']] == label]
            # convert GeoDataFrame into GeoJSON object
            geo = layer.__geo_interface__

            # initialize iterator for output image suffix
            count = 0
            # loop through features
            for feature in geo['features']:
                # store relevant information about the feature in focus
                ftype = feature['geometry']['type']
                polygons = feature['geometry']['coordinates']
                bbox = feature['bbox']

                # calculate dimension ratios to be used for sizing output image
                # https://gist.github.com/ianfieldhouse/2284557
                xdist = bbox[2] - bbox[0]
                ydist = bbox[3] - bbox[1]
                xratio = new_dimension / xdist
                yratio = new_dimension / ydist

                # initialize objects for creating a new output image
                size = (self.params['dimension'], self.params['dimension'])
                img = Image.new(mode=self.params['mode'], size=size, color='#ffffff')
                draw = ImageDraw.Draw(img)
                pixels = []

                # generate scaled image representation of feature polygon(s) in focus
                # https://gist.github.com/ianfieldhouse/2284557
                if ftype == 'Polygon':
                    for coords in polygons[0]:
                        px = int(self.params['dimension'] - (bbox[2] - coords[0]) * xratio)
                        py = int((bbox[3] - coords[1]) * yratio)
                        pixels.append((px, py))
                    draw.polygon(pixels, outline='#000000', fill=self.params['fill'])
                elif ftype == 'MultiPolygon':
                    for polygon in polygons:
                        for coords in polygon[0]:
                            px = int(self.params['dimension'] - (bbox[2] - coords[0]) * xratio)
                            py = int((bbox[3] - coords[1]) * yratio)
                            pixels.append((px, py))
                        draw.polygon(pixels, outline='#000000', fill=self.params['fill'])
                        pixels = []

                # apply padding to newly drawn image
                # https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
                new_img = Image.new(mode='1', size=(new_dimension, new_dimension), color='#ffffff')
                inner_x = (new_dimension - self.params['dimension']) // 2
                inner_y = (new_dimension - self.params['dimension']) // 2
                new_img.paste(img, (inner_x, inner_y))

                # save newly drawn image
                new_img.save('data/images/' + label + '/' + label + '_' + str(count) + '.png')
                # increment output image suffix
                count += 1

    # purpose: split image data into train/validation/test sets
    def split_data(self):
        root = 'data/images/'
        # assign list of class labels
        dirs = os.listdir(root)
        # create list of class labels lacking minimum number of observations to be usable
        others = [name for name in dirs if len(os.listdir(root + name)) < self.params['min_class_count']]

        # initialize parallel storage for input and target label data
        x = []
        y = []

        # loop through class labels
        for i in dirs:
            images = os.listdir(root + i)
            # use 'OTHR' label if class label in focus lacks minimum number of observations to be usable
            label = 'OTHR' if i in others else i

            # populate input and target label storage
            for image in images:
                x.append(image)
                y.append(label)

        # convert lists of data to arrays
        x = np.array(x)
        y = np.array(y)

        # generate train/validation/test data splits
        tsize = self.params['test_size']
        vsize = self.params['validation_size']
        xtrain, xvalid, ytrain, yvalid = train_test_split(x, y, test_size=vsize, random_state=0)
        xtrain, xtest, ytrain, ytest = train_test_split(xtrain, ytrain, test_size=tsize, random_state=0)

        # oversample training set
        if self.params['oversample']:
            ros = RandomOverSampler(random_state=0)
            xtrain = xtrain.reshape(-1, 1)
            xtrain, ytrain = ros.fit_resample(xtrain, ytrain)
            xtrain = xtrain.reshape(-1)

        # rebuild list of class labels
        dirs = [i for i in dirs if i not in others]
        if len(others) > 0:
            dirs.append('OTHR')

        # move image splits to their appropriate destination
        self.move_images('train/', dirs, xtrain, ytrain)
        self.move_images('validation/', dirs, xvalid, yvalid)
        self.move_images('test/', dirs, xtest, ytest)

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
            # https://stackoverflow.com/questions/33282647/python-shutil-copy-if-i-have-a-duplicate-file-will-it-copy-to-new-location
            if not os.path.exists(destination):
                shutil.copyfile(source, destination)
            else:
                base, extension = os.path.splitext(x[i])
                count = 1
                while os.path.exists(destination):
                    destination = 'data/' + path + y[i] + '/' + base + '_' + str(count) + extension
                    count += 1
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
