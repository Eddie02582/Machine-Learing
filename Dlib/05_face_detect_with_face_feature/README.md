# Face Detect with Face Feature

```python
import dlib
import cv2
import imutils
import os
import sys

os.chdir(sys.path[0])

# Ū������
img = cv2.imread('image.jpg')

# �Y�p�Ϥ�
img = imutils.resize(img, width=800)

# Dlib ���H�y������
detector = dlib.get_frontal_face_detector()

#�H�y68�S�x�I�ҫ��˴���
shape_predictor = dlib.shape_predictor("..\dat\shape_predictor_68_face_landmarks.dat") 

# �����H�y
face_rects = detector(img, 0)

# ���X�Ҧ����������G
for i, d in enumerate(face_rects):    
    x1 = d.left()
    y1 = d.top()
    x2 = d.right()
    y2 = d.bottom()
    
    #68�S�x�I����
    shape = shape_predictor(img,d)
    for index, pt in enumerate(shape.parts()):
        print('Part {}: {}'.format(index, pt))
        pt_pos = (pt.x, pt.y)
        cv2.circle(img, pt_pos, 2, (255, 0, 0), 1)

# ��ܵ��G
cv2.imshow("Face Detection", img)

cv2.waitKey(0)
cv2.destroyAllWindows()

```
