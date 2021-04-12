# Face Detect


```python
import dlib
import cv2
import imutils
import os
import sys

os.chdir(sys.path[0])

# Ū������
img = cv2.imread('image.jpg')

#open cv is bgr =>need to change rgb


# �Y�p�Ϥ�
img = imutils.resize(img, width=800)

#open cv is bgr =>need to change rgb
img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

# Dlib ���H�y������
detector = dlib.get_frontal_face_detector()

# �����H�y
face_rects = detector(img_rgb, 0)

# ���X�Ҧ����������G
for i, d in enumerate(face_rects):
    x1 = d.left()
    y1 = d.top()
    x2 = d.right()
    y2 = d.bottom()
    # �H��ؼХܰ������H�y
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)


# ��ܵ��G
cv2.imshow("Face Detection", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
```
