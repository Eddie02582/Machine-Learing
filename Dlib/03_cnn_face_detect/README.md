# Face Detect used CNN

�H�y�˴��@�ˮt�O�O�ϥ�CNN model

```python
import dlib
from skimage import io
import cv2
import imutils
import os
import sys

os.chdir(sys.path[0])


#Ū������
#img = io.imread('image.jpg')
img = cv2.imread('image.jpg')

# �Y�p�Ϥ�
img = imutils.resize(img, width=800)

# cnn ���H�y������
cnn_face_detector = dlib.cnn_face_detection_model_v1('..\dat\mmod_human_face_detector.dat')

# �����H�y
face_rects = cnn_face_detector(img, 0)



# ���X�Ҧ����������G
for i, d in enumerate(face_rects):
    
    x1 = d.rect.left()
    y1 = d.rect.top()
    x2 = d.rect.right()
    y2 = d.rect.bottom()   

    # �H��ؼХܰ������H�y
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)
   
# ��ܵ��G
cv2.imshow("Face Detection", img)

cv2.waitKey(0)
cv2.destroyAllWindows()



```
