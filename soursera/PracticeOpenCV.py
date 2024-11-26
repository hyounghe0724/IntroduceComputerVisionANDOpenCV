import cv2
import matplotlib.pyplot as plt
image =  cv2.imread("../lenna.png")
print(type(image))
print(image.shape)
print(image.max)
print(image.min)

plt.figure(figsize = (10,10))
plt.imshow(image)
#plt.show()

new_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize = (10,10))
plt.imshow(image)
#plt.show()

#image save
#cv2.imwrite("lenna.jpg", image)

"""
GrayScale Images
"""
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(image_gray.shape)

plt.figure(figsize = (10,10))
plt.imshow(image_gray, cmap='gray')
#plt.show()
# save gray scale image  as jpg as well,
# cv2.imwrite('lena_gray_cv.jpg',  image_gray)
im_gray = cv2.imread('../barbara.png', cv2.IMREAD_GRAYSCALE)

plt.figure(figsize = (10,10))
plt.imshow(im_gray, cmap='gray')
#plt.show()

"""
Color Channels
"""

baboon = cv2.imread("../baboon.png")
plt.figure(figsize = (10,10))
plt.imshow(cv2.cvtColor(baboon, cv2.COLOR_BGR2RGB))
plt.show()

blue, green, red = baboon[:,:,0], baboon[:,:,1], baboon[:,:,2]

im_bgr = cv2.vconcat([blue, green, red])

plt.figure(figsize = (10,10))
plt.subplot(121)
plt.imshow(cv2.cvtColor(baboon, cv2.COLOR_BGR2RGB))
plt.title("RGB image")
plt.subplot(122)
plt.imshow(im_bgr, cmap='gray')
plt.title("Different color channels blue (top), green (middle), red (bottom)")
plt.show()

"""
Indexing
"""

rows = 256
plt.figure(figsize = (10, 10))
plt.imshow(new_image[0:rows, :, :])
plt.show()

column = 256
plt.figure(figsize = (10, 10))
plt.imshow(new_image[:,0:column,:])
plt.show()

A = new_image.copy()
plt.imshow(A)
plt.show()

B = A
A[:,:,:] = 0
plt.imshow(B)
plt.show()

baboon_red = baboon.copy()
baboon_red[:,:,0] = 0
baboon_red[:,:,1] = 0
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(baboon_red, cv2.COLOR_BGR2RGB))
plt.show()

baboon_green = baboon.copy()
baboon_green[:,:,0] = 0
baboon_green[:,:,2] = 0
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(baboon_green, cv2.COLOR_BGR2RGB))
plt.show()

image = cv2.imread("../baboon.png")

baboon_blue = image.copy()
baboon_blue[:,:,1] = 0
baboon_blue[:,:,2] = 0
plt.figure(figsize = (10, 10))
plt.imshow(cv2.cvtColor(baboon_blue, cv2.COLOR_BGR2RGB))
plt.show()

"""
Question 1
"""

baboon_blue=cv2.imread('../baboon.png')
baboon_blue=cv2.cvtColor(baboon_blue, cv2.COLOR_BGR2RGB)
baboon_blue[:,:,2] = 0
plt.figure(figsize=(10,10))
plt.imshow(baboon_blue)
plt.show()