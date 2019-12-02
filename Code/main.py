########################################################################################################################
# LOAD DEPENDENCIES
########################################################################################################################

import init

########################################################################################################################
# PROJECT PREPARATION
########################################################################################################################

# path to shapefile of interest
SHAPEFILE = 'data/shapefiles/CONUS/Species_CONUS_Range_2001v1.shp'
# name of feature to use for class labels
TARGET = 'Taxa'

# desired dimension of output images minus padding size
DIMENSION = 240
# padding size for images
PADDING = 10
# minimum count of observations for valid class label
MIN_COUNT = 1
# validation/test split sizes
VALID_SIZE = 0.2
TEST_SIZE = 0.1

# run project preparations
dataset = init.Dataset(SHAPEFILE, TARGET, DIMENSION, PADDING, MIN_COUNT, VALID_SIZE, TEST_SIZE)
dataset.run()
