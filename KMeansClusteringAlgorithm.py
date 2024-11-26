# K 평균 군집화 알고리즘
import cv2
import numpy as np

src = cv2.imread("egg.jpg")
print(src.shape)
data = src.reshape(-1, 3).astype(np.float32)

K = 13
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TermCriteria_EPS, 10, 0.001)
retval, best_labels, centers = cv2.kmeans(
    data # input data
    ,K # 군집의 갯수
    , None
    , criteria # 기준
    , 10 # 시도
    , cv2.KMEANS_RANDOM_CENTERS
)

centers = centers.astype(np.uint8)
dst = centers[best_labels].reshape(src.shape)

cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destoryAllWindows()


