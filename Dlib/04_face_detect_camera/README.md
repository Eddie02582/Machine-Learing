# Face Detect with Camera

```
import dlib
import cv2


# 開啟攝影機
cap = cv2.VideoCapture(0)

# 使用 XVID 編碼
fourcc = cv2.VideoWriter_fourcc(*'XVID')

width,height = 640,480

# 建立 VideoWriter 物件，輸出影片至 output.avi，FPS 值為 20.0
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (width, height))


# Dlib 的人臉偵測器
detector = dlib.get_frontal_face_detector()

while cap.isOpened():
    ret, frame = cap.read()
    # 偵測人臉
    face_rects, scores, idx = detector.run(frame, 0)    
    for i, d in enumerate(face_rects):
        x1 = d.left()
        y1 = d.top()
        x2 = d.right()
        y2 = d.bottom()

        # 以方框標示偵測的人臉
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)
    
    # 寫入影格
    out.write(frame)
    
    # 顯示結果
    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;


cap.release()
out.release()
cv2.destroyAllWindows()

```
