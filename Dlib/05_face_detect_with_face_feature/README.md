# Face Detect with Face Feature

```python
import dlib
import cv2
import imutils
import os
import sys

os.chdir(sys.path[0])

# 讀取圖檔
img = cv2.imread('image.jpg')

# 縮小圖片
img = imutils.resize(img, width=800)

# Dlib 的人臉偵測器
detector = dlib.get_frontal_face_detector()

#人臉68特徵點模型檢測器
shape_predictor = dlib.shape_predictor("..\dat\shape_predictor_68_face_landmarks.dat") 

# 偵測人臉
face_rects = detector(img, 0)

# 取出所有偵測的結果
for i, d in enumerate(face_rects):    
    x1 = d.left()
    y1 = d.top()
    x2 = d.right()
    y2 = d.bottom()
    
    #68特徵點偵測
    shape = shape_predictor(img,d)
    for index, pt in enumerate(shape.parts()):
        print('Part {}: {}'.format(index, pt))
        pt_pos = (pt.x, pt.y)
        cv2.circle(img, pt_pos, 2, (255, 0, 0), 1)

# 顯示結果
cv2.imshow("Face Detection", img)

cv2.waitKey(0)
cv2.destroyAllWindows()

```
