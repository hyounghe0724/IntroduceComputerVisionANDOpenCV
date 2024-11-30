from shutil import copy2

import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
from PIL import Image
from PIL import ImageOps
import numpy as np
from PIL import ImageDraw
from PIL import ImageFont

"""
Copying Images

baboon = np.array(Image.open("../baboon.png"))
# plt.figure(figsize = (5,5))
# plt.imshow(baboon)
# plt.show()

A = baboon

print(id(A) == id(baboon))

B = baboon.copy()
print(id(B) == id(baboon))

# different memory B with A and baboon
baboon[:,:,:] = 0

plt.figure(figsize(10, 10))
plt.subplot(121)
plt.imshow(baboon)
plt.title("Baboon")
# plt.subplot(122)
# plt.imshow(A)
# plt.title("array A")
plt.subplot(122)
plt.imshow(B)
plt.title("array B")
plt.show()
"""

"""
Flipping Images

image = Image.open("../cat.png")

crop_image = array[upper: lower, left:right, :]
array = np.array(image)
width, height, C =array.shape
print("width, height, C", width, height, C)

# copying to array_flip exchanged original first rows and last rows
array_flip = np.zeros((width, height, C), dtype = np.uint8)

for i,rows in enumerate(array):
    array_flip[width - 1 - i, :, :]  = rows

#im_flip = ImageOps.flip(image)
im_flip = image.transpose(1)
plt.figure(figsize = (5,5))
plt.imshow(im_flip)
plt.show()

flip = {"FLIP_LEFT_RIGHT": Image.FLIP_LEFT_RIGHT,
        "FLIP_TOP_BOTTOM": Image.FLIP_TOP_BOTTOM,
        "ROTATE_90": Image.ROTATE_90,
        "ROTATE_180": Image.ROTATE_180,
        "ROTATE_270": Image.ROTATE_270,
        "TRANSPOSE": Image.TRANSPOSE,
        "TRANSVERSE": Image.TRANSVERSE}

for key, values in flip.items():
    plt.figure(figsize = (10, 10))
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.title("Original")
    plt.subplot(1,2,2)
    plt.imshow(image.transpose(values))
    plt.title(key)
    plt.show()
"""

"""
Cropping an Image

upper = 150 #first row we include in the image
lower = 450 #last row we include in the image

left = 150
right = 400
plt.figure(figsize = (5,5))
plt.imshow(crop_image)
plt.show()

crop_image = crop_image.transpose(Image.FLIP_LEFT_RIGHT)
print(crop_image)
"""

"""
Changing Specific Image Pixel

image = Image.open("../cat.png")
array = np.array(image)
array_sq = np.copy(array)
upper = 150 #first row we include in the image
lower = 450 #last row we include in the image
left = 150
right = 400
array_sq[upper: lower, left:right, 1:3] = 0

plt.figure(figsize = (5,5))
plt.subplot(1,2,1)
plt.imshow(array)
plt.title("Original")
plt.subplot(1,2,2)
plt.imshow(array_sq)
plt.title("Altered Image")
plt.show()

image_draw = image.copy()

image_fn = ImageDraw.Draw(im = image_draw)

shape = [left, upper, right, lower]
image_fn.rectangle(xy = shape, fill = "red")

plt.figure(figsize = (10, 10))
plt.imshow(image_draw)
plt.show()

image_fn.text(xy=(0,0),text="box", fill=(0,0,0))
plt.figure(figsize=(10,10))
plt.imshow(image_draw)
plt.show()

image_lenna = Image.open("../lenna.png")
array_lenna = np.array(image_lenna)
array_lenna[upper:lower, left:right,:] = array[upper:lower,left:right,:]
plt.imshow(array_lenna)
plt.show()

#image_lenna.paste(crop_image, box=(left,upper))
#plt.imshow(image_lenna)
#plt.show()

# image applies to some PIL objects
image = Image.open("../cat.png")
new_image = image
copy_image = image.copy()
print(id(image) == id(copy_image))

image_fn = ImageDraw.Draw(im = image)
image_fn.text(xy=(0,0), text="box", fill = (0,0,0))
image_fn.rectangle(xy=shape, fill = "red")


plt.figure(figsize = (10,10))
plt.subplot(121)
plt.imshow(new_image)
plt.subplot(122)
plt.imshow(copy_image)
plt.show()
"""


"""
Question1

use the image babboon.png from this lab or take my image you like.


"""


