# Face Detect


```python
import dlib
import cv2
import imutils
import os
import sys

os.chdir(sys.path[0])

# 讀取圖檔
img = cv2.imread('image.jpg')

#open cv is bgr =>need to change rgb


# 縮小圖片
img = imutils.resize(img, width=800)

#open cv is bgr =>need to change rgb
img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

# Dlib 的人臉偵測器
detector = dlib.get_frontal_face_detector()

# 偵測人臉
face_rects = detector(img_rgb, 0)

# 取出所有偵測的結果
for i, d in enumerate(face_rects):
    x1 = d.left()
    y1 = d.top()
    x2 = d.right()
    y2 = d.bottom()
    # 以方框標示偵測的人臉
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)


# 顯示結果
cv2.imshow("Face Detection", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
```
