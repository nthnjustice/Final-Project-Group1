import scipy.ndimage
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import PIL.ImageOps

data = np.array(Image.open('/home/ubuntu/Deep-Learning/Final-Project-Group1/Code/data/train/DIST/DIST_14.png').convert('L'))
# assert data.ndim == 2, "Image must be monochromatic"
# finds and number all disjoint white regions of the image
is_white = data > 128
labels, n = scipy.ndimage.measurements.label(is_white)

# get a set of all the region ids which are on the edge - we should not fill these
on_border = set(labels[:, 0]) | set(labels[:, -1]) | set(labels[0, :]) | set(labels[-1, :])

for label in range(1, n+1):  # label 0 is all the black pixels
    if label not in on_border:
        # turn every pixel with that label to black
        data[labels == label] = 0

img = Image.fromarray(np.uint8(data*255))
# img_inv = PIL.ImageOps.invert(img)
# img_border = PIL.ImageOps.expand(img_inv, border=10, fill='white')
plt.imsave('test.png', img, cmap='Greys')