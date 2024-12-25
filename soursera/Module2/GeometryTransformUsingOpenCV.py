import matplotlib.pyplot as plt
import cv2
import numpy as np

def plot_image(image_1, image_2,title_1="Orignal",title_2="New Image"):
    plt.figure(figsize=(10,10))
    plt.subplot(1, 2, 1)
    plt.imshow(image_1,cmap="gray")
    plt.title(title_1)
    plt.subplot(1, 2, 2)
    plt.imshow(image_2,cmap="gray")
    plt.title(title_2)
    plt.show()


"""
Geometric Transformation
"""
toy_image = np.zeros((6,6))
toy_image[1:5, 1:5] = 255
toy_image[2:4, 2:4] = 0
plt.imshow(toy_image, cmap="gray")
plt.show()


# fx : scale factor along the horizontal axis
# fy : scale factor along the vertical axis

"""
The parameter interpolation estimates pixel values based on neighboring pixels.
INTER_NEAREST uses the nearest pixel and
 INTER_CUBIC uses several pixels near the pixel value we would like to estimate.
"""

#new_toy = cv2.resize(toy_image, None, fx=2, fy=1, interpolation=cv2.INTER_NEAREST)
new_toy = cv2.resize(toy_image, None, fx=2, fy=1, interpolation=cv2.INTER_CUBIC)
plt.imshow(new_toy, cmap="gray")
plt.show()

image = cv2.imread("/soursera/Module2/lenna.png")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

new_image = cv2.resize(image, None, fx=2, fy=1, interpolation=cv2.INTER_CUBIC)
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()
print("old image shape:", image.shape, "new image shape:", new_image.shape)

new_image = cv2.resize(image, None, fx=1, fy=2, interpolation=cv2.INTER_CUBIC)
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()
print("old image shape:", image.shape, "new image shape:", new_image.shape)

new_image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()
print("old image shape:", image.shape, "new image shape:", new_image.shape)

new_image = cv2.resize(image, None, fx=1, fy=0.5, interpolation=cv2.INTER_CUBIC)
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()
print("old image shape:", image.shape, "new image shape:", new_image.shape)

rows = 100
cols = 200
new_image = cv2.resize(image,(100, 200), interpolation=cv2.INTER_CUBIC)
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()
print("old image shape:", image.shape, "new Image shape:", new_image.shape)

"""
Translation
"""
tx = 100
ty = 0
M = np.float32([[1, 0, tx], [0, 1, ty]])
print(M)
"""
cv2.warpAffine

first param : image array
second param : input parameter is the transformation matrix M
final param : length and width of the output image ( cols, rows )
"""

new_image = cv2.warpAffine(image, M, (cols, rows))
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()

new_image = cv2.warpAffine(image, M, (cols + tx, rows + ty))
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()

# shift image horizontally

# rows = 100
# cols = 200
tx = 0
ty = 50
M = np.float32([[1, 0, tx], [0, 1, ty]])
#원본 이미지가 50만쿰 y축으로 이동 했으므로,
#윈본 이미지 전체를 띄우기 위해 height에 이동한 만큼 더해서 plot을 띄움
new_image = cv2.warpAffine(image, M, (cols+ tx,rows+ ty)) # 200, 150
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()
