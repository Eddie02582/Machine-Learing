# Face Detect With Score

�P�H�y�˴��@�ˮt�O�O�ϥ�detector.run���Ndetector

```
import dlib
import cv2
import imutils

# Ū������
img = cv2.imread('image.jpg')

# �Y�p�Ϥ�
img = imutils.resize(img, width=800)

# Dlib ���H�y������
detector = dlib.get_frontal_face_detector()

# �����H�y
#face_rects = detector(img, 0)

# �����H�y�A��X����
#The third argument to run is an optional adjustment to the detection threshold,
# where a negative value will return more detections and a positive value fewer.
face_rects, scores, idx = detector.run(img, 0, -1)


# ���X�Ҧ����������G
for i, d in enumerate(face_rects):
    x1 = d.left()
    y1 = d.top()
    x2 = d.right()
    y2 = d.bottom()
    text = "%2.2f(%d)" % (scores[i], idx[i])

    # �H��ؼХܰ������H�y
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (x1, y1), cv2.FONT_HERSHEY_DUPLEX,0.7, (255, 255, 255), 1, cv2.LINE_AA)
# ��ܵ��G
cv2.imshow("Face Detection", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
```
