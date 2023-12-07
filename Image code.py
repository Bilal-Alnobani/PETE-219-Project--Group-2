##### Image Analysis code goes here: #####

#### importing libraries
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#### Viewing image
image = Image.open('berea8bit.tif')
image.show()
imarray = np.array(image)

img = io.imread('berea8bit.tif')
plt.imshow(img,cmap='gray')
plt.axis('off')
plt.show()
imarray = np.array(img)

#### displaying image as a histogram
# convert image to a property vector
imV = imarray.reshape((500*500, 1))
# histogram
plt.hist(imV, density=True, bins=30, range=[120,270]) 
plt.show()

### binary image thresholding
BW = imarray
BW[BW<100] = 0
BW[BW>=100] = 255
BW2 = BW.astype(bool)
BW2 = np.array(BW2)
plt.imshow(imarray, cmap='Greys_r')
plt.axis('off')

area = np.size(BW2)
fw = np.sum(BW2)

print('The porosity of the thin section is:', (1 - (fw/area))*100)
