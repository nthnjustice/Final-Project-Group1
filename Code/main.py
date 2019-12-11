# runs everything to reproduce final project efforts

from spatial_data_setup.setup import setup
from models.nj import nj
from models.dv import dv
from models.pre_train_model import pre_train_model

# initialize project assetts
setup()
# run custom CNN models
nj()
dv()
# run pre-trained CNN model
pre_train_model()
