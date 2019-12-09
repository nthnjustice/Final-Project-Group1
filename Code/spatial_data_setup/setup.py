from utils.get_datainfo import get_datainfo
from utils.init_dir import init_dir
from spatial_data_setup.fetch_data import fetch_data
from spatial_data_setup.sp_2_png import sp_2_png
from spatial_data_setup.split_data import split_data

# data_source = 'neighborhoods'
img_dim = 100
img_mode = '1'
img_fill = '#000000'
min_obsv = 1
valid_size = 0.2
test_size = 0.1
oversample = True

init_dir('data')
init_dir('data/spatial')
name, url = get_datainfo(data_source)
print('fetching data from remote source...')
fetch_data(name, url, 'data/spatial')

init_dir('data/images')
print('converting shapes into images...')
sp_2_png(img_dim, img_mode, img_fill)

init_dir('data/train')
init_dir('data/validation')
init_dir('data/test')
print('building train/validation/test data splits...')
split_data(min_obsv, valid_size, test_size, oversample)
