########################################################################################################################
# LOAD DEPENDENCIES
########################################################################################################################

from dataset import Dataset

########################################################################################################################
# PROJECT PREPARATION
########################################################################################################################

params = {
    # names of data-related directories to be created
    'data_dirs': ['data', 'images', 'shapefiles', 'test', 'train', 'validation', 'models'],
    # path to file with remote data-source names and locations
    'sources': 'sources.txt',
    # path to shapefile of interest (available after .run() call)
    'shapefile': 'data/shapefiles/CONUS/Species_CONUS_Range_2001v1.shp',
    # name of feature to use for class labels
    'target': 'Taxa',
    # pixel mode of output images
    'mode': '1',
    # desired dimension of output images minus padding size
    'dimension': 240,
    # padding size for images
    'padding': 10,
    # polygon fill color (white = #ffffff, black = #000000)
    'fill': '#000000',
    # boolean value for random oversampling the training set
    'oversample': False,
    # minimum number of observations for usable class label
    'min_class_count': 1,
    # validation split size
    'validation_size': 0.2,
    # test split size
    'test_size': 0.1
}

# run project preparations
dataset = Dataset(params)
dataset.run()

