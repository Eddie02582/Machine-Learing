# Face Detect with Camera

```
import dlib
import cv2


# �}����v��
cap = cv2.VideoCapture(0)

# �ϥ� XVID �s�X
fourcc = cv2.VideoWriter_fourcc(*'XVID')

width,height = 640,480

# �إ� VideoWriter ����A��X�v���� output.avi�AFPS �Ȭ� 20.0
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (width, height))


# Dlib ���H�y������
detector = dlib.get_frontal_face_detector()

while cap.isOpened():
    ret, frame = cap.read()
    # �����H�y
    face_rects, scores, idx = detector.run(frame, 0)    
    for i, d in enumerate(face_rects):
        x1 = d.left()
        y1 = d.top()
        x2 = d.right()
        y2 = d.bottom()

        # �H��ؼХܰ������H�y
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)
    
    # �g�J�v��
    out.write(frame)
    
    # ��ܵ��G
    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;


cap.release()
out.release()
cv2.destroyAllWindows()

```
