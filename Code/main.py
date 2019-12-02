import init

SHAPEFILE = 'data/shapefiles/CONUS/Species_CONUS_Range_2001v1.shp'
TARGET = 'Taxa'

DIMENSION = 240
PADDING = 10
MIN_COUNT = 1
TEST_SIZE = 0.2
VALID_SIZE = 0.2

dataset = init.Dataset(SHAPEFILE, TARGET, DIMENSION, PADDING, MIN_COUNT, TEST_SIZE, VALID_SIZE)
dataset.run()
