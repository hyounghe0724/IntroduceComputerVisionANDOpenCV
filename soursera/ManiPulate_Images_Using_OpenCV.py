import cv2
import matplotlib.pyplot as plt
import numpy as np

from soursera.ManipulatingImagesUsingPillow import im_flip

"""
Copying Images

baboon = cv2.imread("../baboon.png")
plt.figure(figsize = (10,10))
plt.imshow(cv2.cvtColor(baboon, cv2.COLOR_BGR2RGB))
plt.show()

A = baboon

print(id(A) == id(baboon))
B = baboon.copy()
print(id(B) == id(baboon))

baboon[:,:,:] = 0

plt.figure(figsize=  (10, 10))
plt.subplot(121)
plt.imshow(cv2.cvtColor(baboon, cv2.COLOR_BGR2RGB))
plt.title("babbon")
plt.subplot(122)
plt.imshow(cv2.cvtColor(A, cv2.COLOR_BGR2RGB))
plt.title("array A")
plt.show()

plt.figure(figsize  = (10, 10))
plt.subplot(121)
plt.imshow(cv2.cvtColor(baboon, cv2.COLOR_BGR2RGB))
plt.title("baboon")
plt.subplot(122)
plt.imshow(cv2.cvtColor(B, cv2.COLOR_BGR2RGB))
plt.title("array B")
plt.show()
"""

"""
Fliping Images

image = cv2.imread("../cat.png")
plt.figure(figsize = (10, 10))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()


width, height, C = image.shape
print("width, height, C", width, height, C)

array_flip = np.zeros((width, height,C), dtype=np.uint8)
# same shape original array, this array values is zeros
# last row of pixel of original to new array first of pixel
# , We repeat the process every row to new array
for i, row in enumerate(image):
    array_flip[width-1-i,:,:] = row
#result
plt.figure(figsize = (5,5))
plt.imshow(cv2.cvtColor(array_flip, cv2.COLOR_BGR2RGB))
plt.show()

# OpenCV serve fliping function with flipcode
# flipcode = 0: X
# flipcode  > 0 : Y
# flipcode < 0 : X and y
for flipcode in [0, 1, -1]:
    im_flip = cv2.flip(image, flipcode)
    plt.imshow(cv2.cvtColor(im_flip, cv2.COLOR_BGR2RGB))
    plt.title("flipcode: " + str(flipcode))
    plt.show()
im_flip = cv2.rotate(image, 0)
plt.imshow(cv2.cvtColor(im_flip, cv2.COLOR_BGR2RGB))
plt.show()

flip = {"ROTATE_90_CLOCKWISE":cv2.ROTATE_90_CLOCKWISE
    ,"ROTATE_90_COUNTERCLOCKWISE":cv2.ROTATE_90_COUNTERCLOCKWISE
    ,"ROTATE_180":cv2.ROTATE_180}

print(flip["ROTATE_90_CLOCKWISE"])

for key, value in flip.items():
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.subplot(1,2,2)
    plt.imshow(cv2.cvtColor(cv2.rotate(image,value), cv2.COLOR_BGR2RGB))
    plt.title(key)
    plt.show()
"""

"""
Cropping an Image

image = cv2.imread("../cat.png")
upper = 150
lower = 400
crop_top = image[upper: lower, :, :]
plt.figure(figsize = (10,10))
plt.imshow(cv2.cvtColor(crop_top, cv2.COLOR_BGR2RGB))
plt.show()

left = 150
right = 400
crop_horizontal = crop_top[:, left:right, :]
plt.figure(figsize= (5,5))
plt.imshow(cv2.cvtColor(crop_horizontal, cv2.COLOR_BGR2RGB))
plt.show()
"""

"""
Changing Specific Image Pixel
upper = 150
lower = 400
left = 150
right = 400

image = cv2.imread("../cat.png")
array_np = np.copy(image)
array_np[upper:lower, left:right,:] = 0

plt.figure(figsize = (10,10))
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("original")
plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(array_np, cv2.COLOR_BGR2RGB))
plt.title("Altered Image")
plt.show()

start_point, end_point = (left,upper),(right,lower)
image_draw = np.copy(image)
cv2.rectangle(image_draw, start_point, end_point, (0, 255, 0), thickness=3)
plt.figure(figsize = (5,5))
plt.imshow(cv2.cvtColor(image_draw, cv2.COLOR_BGR2RGB))
plt.show()

plt.figure(figsize = (10,10))
plt.show()
image_draw = cv2.putText(img = image
                         , text="Stuff"
                         , org=(10, 500)
                         , color= (255,255,255)
                         ,fontFace = 4
                         ,fontScale =5
                         , thickness=3)
plt.figure(figsize = (10,10))
plt.imshow(image_draw)
plt.show()
"""



"""
Question 4

"""

im = cv2.imread("../cat.png")
cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
im_flip = cv2.flip(im, 0)
im_mirror = cv2.flip(im, 1)

plt.figure(figsize = (10,10))
plt.subplot(1,2,1)
plt.imshow(im_flip)
plt.subplot(1,2,2)
plt.imshow(im_mirror)
plt.show()
