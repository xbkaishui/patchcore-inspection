import cv2
import os
from PIL import Image
import matplotlib.image as mpimg
import numpy as np

image = mpimg.imread('/opt/.pc/mvtec/bottom/test/NG/0029-B.jpg')
image.flags.writeable = True  # 将数组改为读写模式`
Image.fromarray(np.uint8(image))
mask = mpimg.imread('/tmp/a.png')
image[:,:,:][mask[:,:,:]>0] = 255
cv2.imwrite('test.png',image)