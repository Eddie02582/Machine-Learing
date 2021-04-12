# Face Detect used CNN

人臉檢測一樣差別是使用CNN model

```python
import dlib
from skimage import io
import cv2
import imutils
import os
import sys

os.chdir(sys.path[0])


#讀取圖檔
#img = io.imread('image.jpg')
img = cv2.imread('image.jpg')

# 縮小圖片
img = imutils.resize(img, width=800)

# cnn 的人臉偵測器
cnn_face_detector = dlib.cnn_face_detection_model_v1('..\dat\mmod_human_face_detector.dat')

# 偵測人臉
face_rects = cnn_face_detector(img, 0)



# 取出所有偵測的結果
for i, d in enumerate(face_rects):
    
    x1 = d.rect.left()
    y1 = d.rect.top()
    x2 = d.rect.right()
    y2 = d.rect.bottom()   

    # 以方框標示偵測的人臉
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)
   
# 顯示結果
cv2.imshow("Face Detection", img)

cv2.waitKey(0)
cv2.destroyAllWindows()



```
