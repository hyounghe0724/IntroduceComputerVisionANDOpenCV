import matplotlib.pyplot as plt
import cv2

import numpy as np

def plot_image(image_1, image_2,title_1="Orignal",title_2="New Image"):
    plt.figure(figsize=(10,10))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB))
    plt.title(title_1)
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB))
    plt.title(title_2)
    plt.show()

image = cv2.imread("lenna.png")

#plt.imshow(cv2.cvtColor(image, cv2.COLOR_YUV2BGR))

rows, cols,_ = image.shape
noise = np.random.normal(0, 15, (rows, cols,3)).astype(np.uint8)
noisy_image = image + noise
#plot_image(image, noisy_image, "Original", "Image Plus Noise")

"""
Filtering Noise
"""

kernel = np.ones((6,6))/36

image_filtered = cv2.filter2D(src=noisy_image,ddepth=-1,kernel=kernel)
#plot_image(image_filtered, noisy_image, "filterd Image", "Image Plus NOise")

"""
Gaussian Blur

"""

image_filtered = cv2.GaussianBlur(noisy_image, (5,5), sigmaX= 4, sigmaY= 4)
#plot_image(image_filtered, noisy_image, "Gaussian filtered image", "Image PLus Noise")
image_filtered = cv2.GaussianBlur(noisy_image, (11,11), sigmaX= 10, sigmaY= 10)
#plot_image(image_filtered, noisy_image, "Gaussian filtered image", "Image PLus Noise")

"""
Image Sharpening
"""
kernel = np.array(
    [[-1,-1,-1],
    [-1,9,-1],
    [-1,-1,-1]]
)
sharpened = cv2.filter2D(image,-1, kernel)
#plot_image(sharpened, image, "Sharpened", "iamge")

"""
Edges
"""

img_gray = cv2.imread("barbara.png", cv2.IMREAD_GRAYSCALE)
print(img_gray)
plt.imshow(img_gray, cmap="gray")

img_gray = cv2.GaussianBlur(img_gray, (3,3), sigmaX= 0.1, sigmaY= 0.1)
plt.imshow(img_gray, cmap="gray")
#plt.show()


# Sobel함수를 이용한 X 또는 Y 방향의 도함수에 근사를 구할 수 있다.

ddepth = cv2.CV_16S
grad_x = cv2.Sobel(src=img_gray, ddepth=ddepth, dx=1, dy=0, ksize=3)
plt.imshow(grad_x, cmap="gray")
grad_y = cv2.Sobel(src=img_gray, ddepth=ddepth, dx=0, dy=1, ksize=3)
plt.imshow(grad_y, cmap="gray")
# 왜 절댓값을 하는 이유?
abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)

#plot_image(abs_grad_x, abs_grad_y, "abs_grad_x", "abs_grad_y")
grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
plt.figure(figsize=(10,10))
plt.imshow(grad, cmap = 'gray')
#plt.show()

"""
Median
"""
image = cv2.imread("cameraman.jpeg", cv2.IMREAD_GRAYSCALE)

filtered_image = cv2.medianBlur(image, 5)
plt.figure(figsize=(10,10))
plt.imshow(filtered_image, cmap='gray')
#plt.show()

"""
threshold function prarmeters
"""

rets, outs = cv2.threshold(src=image, thresh=0, maxval= 255, type = cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
# 여기서 thresh 매개변수는 필요없는 할당만 하는 것
# OTSU가 최적의 임계값을 찾고 임계값 이상은 0 임계값 이하가 maxval ( INVerse라서, 바이너리 함수는 maxval가 255로 고정인듯?)
# 이본 THERSHOLD_BINARY는 임계값이상이 maxval로 초기화되고, 임계값 이하가 0
plt.figure(figsize=(10,10))
plt.imshow(outs, cmap='gray')
plt.show()
