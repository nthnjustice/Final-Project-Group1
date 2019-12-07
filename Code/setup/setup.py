from utils.get_datainfo import get_datainfo
from utils.init_dir import init_dir
from setup.fetch_data import fetch_data
from setup.sp_2_png import sp_2_png
from setup.split_data import split_data

data_source = 'buildings'
target = None
img_dim = 90
img_pad = 10
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
sp_2_png(img_dim, img_pad, img_mode, img_fill, target)

init_dir('data/train')
init_dir('data/validation')
init_dir('data/test')
print('building train/validation/test data splits...')
split_data(min_obsv, valid_size, test_size, oversample)
