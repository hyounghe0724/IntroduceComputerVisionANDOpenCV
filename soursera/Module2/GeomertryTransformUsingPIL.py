import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageOps
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
Geometric Transformations
"""

image = Image.open("/soursera/Module2/lenna.png")
plt.imshow(image)
plt.show()

# horizontal axis by two
width, height = image.size
new_width = 2 * width
new_height = height
new_image = image.resize((new_width, new_height))
plt.imshow(new_image)
plt.show()

# vertical axis by two:

new_width = width
new_height = 2*height
new_image = image.resize((new_width, new_height))
plt.imshow(new_image)
plt.show()

# double both
new_width = 2*width
new_height = 2*height
new_image = image.resize((new_width, new_height))
plt.imshow(new_image)
plt.show()

# shrink both 1/2
new_width = width // 2
new_height = height // 2
new_image = image.resize((new_width, new_height))
plt.imshow(new_image)
plt.show()


"""
Rotation
"""

theta = 45
new_image = image.rotate(theta)
plt.imshow(new_image)
plt.show()

"""
Mathematical Operations
"""

# Array Oprations

image = np.array(image)
# Using Python broadcasting, add constant to each pixel's intensitu value
new_image = image + 20
plt.imshow(new_image)
plt.show()

new_image = 10 * image
plt.imshow(new_image)
plt.show()

# height is image's height
# width is image's width
Noise = np.random.normal(0, 20, (height, width, 3)).astype(np.uint8)
print(Noise.shape)

new_image = image + Noise
plt.imshow(new_image)
plt.show()

"""
Matrix Operations
"""

im_gray = Image.open("D:\ComputerVisionReady\\barbara.png")

im_gray = ImageOps.grayscale(im_gray)

im_gray = np.array(im_gray)
print(im_gray.shape)
plt.imshow(im_gray, cmap="gray")
plt.show()


"""
행렬을 위해 설계된 알고리즘을 적용할 수 있습니다. 
이미지 행렬을 세 개의 행렬의 곱으로 분해하는 특이값 분해 알고리즘을 사용할 수 있습니다.
"""

U,s, V = np.linalg.svd(im_gray, full_matrices=True)
# U : 직교 행렬 mxm크기
# s : 특이값 벡터, 1차원 벡터, 이미지의 중요도
# V : 직교 행렬 nxn 크기
print(s.shape)

# diagonal matrix => 대각선 행렬
S = np.zeros((im_gray.shape[0], im_gray.shape[1])) # width height
S[:image.shape[0], :image.shape[0]] = np.diag(s) # diag() -> s 를 대각 행렬로 변환(S), 이미지 복구 또는 변환 준비 단계
plot_image(U, V, title_1 = "Matrix U,", title_2 = "Matrix V")
plt.imshow(S, cmap="gray")
plt.show()

"""
모든 행렬의 행렬 곱(matrix product)을 구할 수 있다.
S와 U의 행렬 곱셉을 수행하여 B에 할당하고 결과를 플롯할 수 있다.
"""

B = S.dot(V)
plt.imshow(B, cmap="gray")
plt.show()

"""
We can find the matrix product of U and B( B is S and V ). this is entire image
"""
A = U.dot(B)
plt.imshow(A, cmap="gray")
plt.show()

"""
많은 요소들이 중복(redundant)되어 있으므로, S와V의 rows와 columns를 제거(eliminate)하여 원본 이미지에 가깝게 구할 수 있다.

대각 행렬 S의 대부분의 특이값은 작거나 0에 가까움 -> 데이터의 일부가 중요하지 않거나 중복된 정보를 가짐
100개의 특이값이 있을 때, 10개의 특이값이 전체 정보의 90퍼를 설명할 수 있다면, 나머지 90개의 특이값은 사실상 불필요.
"""

# 고유값 분해로 이미지의 복원 품질을 조절한  이미지 플롯
for n_component in [1,10,100,200,500]:
    S_new = S[:, :n_component]
    V_new = V[:n_component, :]
    A = U.dot(S_new.dot(V_new))
    plt.imshow(A,cmap="gray")
    plt.title("Number of Components: {}".format(n_component))
    plt.show()