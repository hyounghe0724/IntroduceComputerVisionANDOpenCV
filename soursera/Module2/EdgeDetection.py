import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from PIL import ImageFilter

def plot_image(image_1, image_2,title_1="Orignal",title_2="New Image"):
    plt.figure(figsize=(10,10))
    plt.subplot(1, 2, 1)
    plt.imshow(image_1)
    plt.title(title_1)
    plt.subplot(1, 2, 2)
    plt.imshow(image_2)
    plt.title(title_2)
    plt.show()


"""
Linear Filtering
"""

image = Image.open("lenna.png")
# # Render the image
# plt.figure(figsize=(5,5))
# plt.imshow(image)
# plt.show()
#
rows, cols = image.size
noise = np.random.normal(0, 15, (rows,cols,3)).astype(np.uint8)
noisy_image = image + noise
noisy_image = Image.fromarray(noisy_image)
# plot_image(image, noisy_image, "Original", "Image Plus Noise")

"""
Filtering Noise

Smoothing filters called low pass filter.
that is average out the Pixels within a neighborhood.
For this filtering, the kernel simply averages out the 
kernels in a neighborhood.

이웃 들의 평균 커널을 구함
"""

kernel = np.ones((5,5),np.uint8)/36

kernel_filter = ImageFilter.Kernel((5,5), kernel.flatten())

"""
함수 filter는 이미지와 커널의 컨볼루션을 수행한다. 각각의 컬러에 대해 독립적으로
"""

image_filtered = noisy_image.filter(kernel_filter)
#plot_image(
#    image_filtered
#   , noisy_image
#    ,"Filtered Image"
#    , "Image Plus Noise"
#)
# 노이스는 감소 되었지만, 이미지가 블러 처리 됨

"""
작은 커널은 이미지는 샤프하게 만들어 주지만, 더 작은 노이스를 필터링한다.
3x3 이미지로 필터링하면 사진속의 그녀는
더 샤프해지지만 초록색 노이즈는 더 밝게 된다.
"""

# Create a kernel which is a 3 by 3 array where each value is 1/36
kernel = np.ones((3,3))/36
# Create a ImageFilter Kernel by providing the kernel size and a flattened kernel
kernel_filter = ImageFilter.Kernel((3,3), kernel.flatten())
# Filters the images using the kernel
image_filtered = noisy_image.filter(kernel_filter)
# Plots the Filtered and Image with Noise using the function defined at the top
#plot_image(image_filtered, noisy_image,title_1="Filtered image",title_2="Image Plus Noise")

"""
Gaussian Blur
"""
# default kernel radius 2
image_filtered = noisy_image.filter(ImageFilter.GaussianBlur)
#plot_image(
 #   image_filtered
 #   ,noisy_image
 #   ,"Filtered Image"
 #   , "Image Plus Noise"
#)

# try using 44
image_filtered = noisy_image.filter(ImageFilter.GaussianBlur(4))
#plot_image(image_filtered, noisy_image, "Filtered Image", "Image Plus Noise")

"""
Image Sharpening

Smoothing과 도함수가 핵심 역할
"""

kernel = np.array([
    [-1,-1,-1]
    ,[-1,9,-1]
    ,[-1,-1,-1]]
)

kernel = ImageFilter.Kernel((3,3), kernel.flatten())
sharpened = image.filter(kernel)
#plot_image(sharpened, image, "Sharpened image", "image")

# sharpen using predefined filter
sharpened = image.filter(ImageFilter.SHARPEN)
#plot_image(sharpened, image, title_1="Sharpened", "image")


"""
Edges 

엣지는 픽셀 밝기가 바뀌는 부분이다. 함수의 그래디언트의 이미지의 변화율이다.
컨볼루션 계산을 통하여 그레이스케일 이미지의 그래디언트를 근사적으로 구할 수 있다
"""

img_gray = Image.open('barbara.png')
#plt.imshow(img_gray, cmap="gray)

# 엣지를 강조시켜 더 나은 엣지를 찾도록 한다.
# Filters the images using EDGE_ENHANCE filter
img_gray = img_gray.filter(ImageFilter.EDGE_ENHANCE)
# Renders the enhanced image
plt.imshow(img_gray ,cmap='gray')

# Filters the images using FIND_EDGES filter
img_gray = img_gray.filter(ImageFilter.FIND_EDGES)
# Renders the filtered image
plt.figure(figsize=(10,10))
plt.imshow(img_gray ,cmap='gray')

#plt.show()

"""
Median

미디안 필터는 kernel안의 모든 픽셀들의 중간 값을 찾는다.
그리고 그 중간값을 이 미디안 값으로 대체한다.
"""

image = Image.open("cameraman.jpeg")
plt.figure(figsize=(10,10,))
plt.imshow(image, cmap='gray')

"""
미디안 필터링은 배경을 블러처리 하고
카메라맨과 배경사이를 좀더 세분화 한다.
"""
image = image.filter(ImageFilter.MedianFilter)
plt.figure(figsize=(10,10))
plt.imshow(image, cmap="gray")
plt.show()