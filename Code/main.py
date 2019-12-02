import init

# SHAPEFILE = 'data/shapefiles/MTPublicLands_SHP/MTPublicLands_SHP.shp'
# TARGET = 'OWNER'
SHAPEFILE = 'data/shapefiles/PADUS2_0_Shapefiles/PADUS2_0Fee.shp'
TARGET = 'Own_Type'

DIMENSION = 240
PADDING = 10
MIN_COUNT = 500
TEST_SIZE = 0.2
VALID_SIZE = 0.2

dataset = init.Dataset(SHAPEFILE, TARGET, DIMENSION, PADDING, MIN_COUNT, TEST_SIZE, VALID_SIZE)
dataset.run()