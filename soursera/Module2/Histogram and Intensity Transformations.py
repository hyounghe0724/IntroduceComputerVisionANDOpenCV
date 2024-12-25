import matplotlib.pyplot as plt
import cv2
import numpy as np



def plot_image(image_1, image_2, title_1 = "Original", title_2 ="New Image"):
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(image_1, cmap="gray")
    plt.title(title_1)
    plt.subplot(1, 2, 2)
    plt.imshow(image_2, cmap="gray")
    plt.title(title_2)
    plt.show()

def plot_hist(old_image, new_image,title_old="Orignal", title_new="New Image"):
    intensity_values=np.array([x for x in range(256)])
    plt.subplot(1, 2, 1)
    plt.bar(intensity_values, cv2.calcHist([old_image],[0],None,[256],[0,256])[:,0],width = 5)
    plt.title(title_old)
    plt.xlabel('intensity')
    plt.subplot(1, 2, 2)
    plt.bar(intensity_values, cv2.calcHist([new_image],[0],None,[256],[0,256])[:,0],width = 5)
    plt.title(title_new)
    plt.xlabel('intensity')
    plt.show()

"""
히스토그램은 픽셀의 빛의 밝기의 값의 갯수를 계산한다. 이미지를 조작하는데 유용한 도구이다.
cv2.calcHist(
 CV array: [image]
 ,this is image channel : [0] # 이미지 채널
 ,for this course it will always be [None]
 ,the number  of bins: [L] # 숫자의 빈도 y
 , the range of index of bins:[0,L-1] # 빈도에 해당하는 인덱스의 범위 x
)
=> x의 범위는 곧 pixel의 밝기, y는 x에 따른 pixel의 밝기 빈도
"""

"""
Toy Example

toy_image = np.array([[0,2,2],[1,1,1],[1,1,2]], dtype=np.uint8)
plt.imshow(toy_image, cmap="gray")
plt.show()
print("toyImage: ",toy_image)

plt.bar([x for x in range(6)], [1,5,2,0,0,0])
plt.show()
plt.bar([x for x in range(6)], [0,1,0,5,0,2])
plt.show()
"""

"""
Gray Scale Histograms

goldhill = cv2.imread("../goldhill.bmp", cv2.IMREAD_GRAYSCALE)

hist = cv2.calcHist([goldhill],[0], None, [256], [0,256])
# hist는 구간 차원의 픽셀 강도에 따른 구간 차원의 픽셀 수를 계산
# 0~20 의 픽셀 수 ->
# 21 ~ .. 의 픽셀 수 -> 
#.. 해서 간략히 표현
intensity_values = np.array([x for x in range(hist.shape[0])])
print(intensity_values.shape)
plt.bar(intensity_values, hist[:,0], width = 5)
plt.title("Bar Histograms")
plt.show()

# 픽셀 수를 정규화하여 확률 질량 함수로 변환할 수 있다.
PMF = hist / (goldhill.shape[0] * goldhill.shape[1])
plt.plot(intensity_values, hist)
plt.title("Histogram")
plt.show()

plt.hist(goldhill.ravel(), 256, [0,256])
plt.show()

# apply a histogram to each image color channel
baboon = cv2.imread("../baboon.png")
plt.imshow(cv2.cvtColor(baboon, cv2.COLOR_BGR2RGB))
plt.show()

color = ('blue', 'green', 'red')
for i, col in enumerate(color):
    histr = cv2.calcHist([baboon], [i], None, [256], [0,256])
    plt.plot(intensity_values # x축 값은 픽셀의 강도
             , histr # 픽셀 강도에서의 빈도 수
             , color=col # 채널의 색으로 그래프를 그림
             , label = col+" channel") # 그래프에 채널이름 표시

    plt.xlim([0,256]) # x 축의 범위를 0~ 256으로 정함
plt.legend() # ㅇ오른쪽 상단에 범례를 표시
plt.title("Histogram Channels")
plt.show()
"""



# 한꺼번에 그려질 수 있는 이유 -> x,y 축과 같은 plot에 새로운 데이터 시리즈를 추가하고,마지막에 plt.show()를 호출하기 때문

"""
image Negatives with Toy example 

toy_image = np.array([[0,2,2],[1,1,1],[1,1,2]], dtype=np.uint8)
plt.imshow(toy_image, cmap="gray")
plt.show()
print("toyImage: ",toy_image)

plt.bar([x for x in range(6)], [1,5,2,0,0,0])
plt.show()
plt.bar([x for x in range(6)], [0,1,0,5,0,2])
plt.show()

neg_toy_image = -1 * toy_image + 255

print("toy image\n", toy_image)
print("image negatives\n", neg_toy_image)

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(toy_image, cmap="gray")
plt.subplot(1,2,2)
plt.imshow(neg_toy_image, cmap="gray")
plt.show()
print("toy_image:", toy_image)

# L = 3인경우
# 0 , 1, 2의 중간 값( 회색은 1 )
# neg = -1 * Pixel + ( 3 - 1)
# neg = -1 * 1 + 2 = 1


image = cv2.imread("../mammogram.png", cv2.IMREAD_GRAYSCALE)
cv2.rectangle(image
              , pt1=(160, 212)
              , pt2=(250,289)
              , color=(0,255,0)
              , thickness=2)
plt.figure(figsize=(10, 10))
plt.imshow(image,cmap="gray")
plt.show()

img_neg = -1 * image + 255
plt.figure(figsize=(10, 10))
plt.imshow(img_neg, cmap="gray")
plt.show()
"""

"""
Brightness and contrast adjustments

goldhill = cv2.imread("../goldhill.bmp", cv2.IMREAD_GRAYSCALE)
alpha = 1
beta = 100
new_image = cv2.convertScaleAbs(goldhill
                                , alpha = alpha
                                , beta = beta
                                )
plot_image(goldhill, new_image
           , title_1 = "Original"
           , title_2 = "brightness control")

plt.figure(figsize=(10, 5))
alpha = 2 # simple contrast control
bata = 0 # simple coontrast control, simple brightness control
new_image = cv2.convertScaleAbs(goldhill
                                ,alpha = alpha
                                , beta = beta
                                )
plot_image(goldhill,new_image,"Original","contrast_control")
plt.figure(figsize=(10, 5))
plot_hist(goldhill, new_image
          ,"Original"
          , "contrast_control")


plt.figure(figsize=(10, 5))
alpha = 3 # contrast control
bate = -200 # brightness control
# contrast maximize, brightness downsizing
new_image = cv2.convertScaleAbs(
    goldhill
    , alpha = alpha
    , beta = bate
)
plot_image(goldhill, new_image
           ,"Original2"
           , "contrast_control"
           )
plt.figure(figsize=(10, 5))
plot_hist(goldhill, new_image
          ,"Original2"
          ,"contrast_control"
          )
"""


"""
Histogram Equalization

is increasing the contrast of images,  그레이 스케일 픽셀의
범위를 확장 하여

zelda = cv2.imread("../zelda.png"
                   ,cv2.IMREAD_GRAYSCALE)
new_image = cv2.equalizeHist(zelda)
plot_image(zelda, new_image
           ,"original"
           , "Histogram equalization")
plt.figure(figsize=(10, 5))
plot_hist(zelda, new_image
          ,"original"
          , "Histogram equalization")

"""


"""
Threshold ( 임계값 ) 
Thresholding And Simple Segmentation

픽셀이 임계값 보다 높으면 zero 또는 255로 설정
"""


def thresholding(input_img, threshold, max_value = 255, min_value = 0):
    N,M = input_img.shape
    image_out = np.zeros((N,M), dtype=np.uint8)
    for i in range(N):
        for j in range(M):
            if input_img[i, j] > threshold:
                image_out[i, j] = max_value
            else:
                image_out[i, j] = min_value
    return image_out


toy_image = np.array([[0,2,2],[1,1,1],[1,1,2]], dtype=np.uint8)
threshold = 1
max_value = 2
min_value = 0
thresholing_toy = thresholding(toy_image
                               , threshold=threshold
                               , max_value=max_value
                               ,min_value=min_value)
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(toy_image, cmap="gray")
plt.title("Original Image")
plt.subplot(1, 2, 2)
plt.imshow(thresholing_toy, cmap="gray")
plt.title("Image After Thresholding")
plt.show()

image = cv2.imread("cameraman.jpeg", cv2.IMREAD_GRAYSCALE)
plt.figure(figsize=(10, 10))
plt.imshow(image, cmap="gray")
plt.show()
hist = cv2.calcHist([image], [0], None, [256], [0,256])
intensity_values = np.array([x for x in range(hist.shape[0])])
plt.bar(intensity_values, hist[:,0], width= 5)
plt.title("bar historgram")
plt.show()

threshold = 87
max_value = 255
min_value = 0
new_image = thresholding(image, threshold=threshold, max_value=max_value, min_value=min_value)

plot_image(image, new_image
           ,"Original"
           ,"Image After Thresholding")
plt.figure(figsize=(10, 10))
plot_hist(image, new_image
        ,"Original"
          ,"Image After Thresholding")

ret, new_image = cv2.threshold(image
                               ,threshold
                               ,max_value
                               ,cv2.THRESH_BINARY)

plot_image(image,new_image,"Orignal","Image After Thresholding")
plot_hist(image, new_image,"Orignal","Image After Thresholding")

# cv2.TRUNC는 픽셀 값이 임계값보다 낮으면 바꾸지 않는다
# 원래는 임게값 보다 크면 바꾸는 것

ret, new_image = cv2.threshold(image
                               ,86
                               ,255
                               , cv2.THRESH_TRUNC )
plot_image(image,new_image,"Orignal","Image After Thresholding")
plot_hist(image, new_image,"Orignal","Image After Thresholding")

ret, otsu = cv2.threshold(image,0,255,cv2.THRESH_OTSU)
plot_image(image,otsu,"Orignal","Otsu")
plot_hist(image, otsu,"Orignal"," Otsu's method")