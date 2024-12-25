from PIL import Image
from PIL import ImageOps
import numpy as np
import matplotlib.pyplot as plt
def get_concat_h(im1, im2):
    #https://note.nkmk.me/en/python-pillow-concat-images/
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst
my_image = "../lenna.png"
image = Image.open(my_image)
#image.show() PIL

im = image.load()
x = 0
y= 1
#print(im[y,x])

#image.save("lenna.png")

"""
GrayScale Images, Quantizaiton and Color Channels
"""

image_gray = ImageOps.grayscale(image)
#image_gray.show()
# print(image_gray.mode)
# image_gray=image_gray.quantize(256 // 4)
# image_gray.show()

"""
for n in range(3, 8):
    plt.figure(figsize=(10, 10))

    plt.imshow(get_concat_h(image_gray, image_gray.quantize(256 // 2**n)))
    plt.title("256 Quantization Levels left vs {} Quantization Levels right".format(256// 2**n))
    plt.show()
"""

"""
Color Channels
"""

baboon = Image.open("../../baboon.png")

red, green, blue = baboon.split()
#get_concat_h(baboon,red).show()
#get_concat_h(baboon,green).show()
#get_concat_h(baboon,blue).show()

array = np.asarray(baboon)
array = np.array(baboon)

# summarize shape
print(array.shape)

#print(array) # pixel intensity, range zero to 255 or 2^8 ( 8-bit)
#print(array[0,0])
#print(array.min(), array.max())

"""
Indexing
"""
# plt.figure(figsize=(10,10))
# plt.imshow(array)
# plt.show()
# rows = 256
#
# plt.figure(figsize=(10,10))
# plt.imshow(array[0:rows, :, :])
# plt.show()
#
# columns = 256
# plt.figure(figsize= (10, 10))
# plt.imshow(array[:,0:columns, :])
# plt.show()

A = array.copy()
# plt.imshow(A)
# plt.show()


# do not apply copy(), variables exist in same location in memory
# So, A to zero, b will be zero too
# B = A
# A[:,:,:] = 0
# plt.imshow(B)
# plt.show()

# baboon_array = np.array(baboon)
# plt.figure(figsize =(10,10))
# plt.imshow(baboon_array)
# plt.show()

# plot the red channel as intensity values of the red channel
baboon_array = np.array(baboon)
# plt.figure(figsize =(10,10))
# plt.imshow(baboon_array[:,:,0], cmap="gray")
# plt.show()

# exclude red channel, others to zero
baboon_red = baboon_array.copy()
baboon_red[:,:,1] = 0
baboon_red[:,:,2] = 0
plt.figure(figsize =(10,10))
plt.imshow(baboon_red)
plt.show()


baboon_blue = baboon_array.copy()
baboon_blue[:,:,0] = 0
baboon_blue[:,:,1] = 0
plt.figure(figsize =(10,10))
plt.imshow(baboon_blue)
plt.show()

lenna = Image.open("lenna.png")
array_lenna = np.array(lenna)
blue_lenna = array_lenna.copy()
blue_lenna[:,:,0] = 0
blue_lenna[:,:,1] = 0
plt.figure(figsize =(10,10))
plt.imshow(blue_lenna)
plt.show()