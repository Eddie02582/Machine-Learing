import dlib
import cv2
import imutils

# 讀取圖檔
img = cv2.imread('image.jpg')

# 縮小圖片
img = imutils.resize(img, width=800)

# Dlib 的人臉偵測器
detector = dlib.get_frontal_face_detector()

# 偵測人臉
#face_rects = detector(img, 0)

# 偵測人臉，輸出分數
#The third argument to run is an optional adjustment to the detection threshold,
# where a negative value will return more detections and a positive value fewer.
face_rects, scores, idx = detector.run(img, 0, -1)


# 取出所有偵測的結果
for i, d in enumerate(face_rects):
    x1 = d.left()
    y1 = d.top()
    x2 = d.right()
    y2 = d.bottom()
    text = "%2.2f(%d)" % (scores[i], idx[i])

    # 以方框標示偵測的人臉
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (x1, y1), cv2.FONT_HERSHEY_DUPLEX,0.7, (255, 255, 255), 1, cv2.LINE_AA)
# 顯示結果
cv2.imshow("Face Detection", img)

cv2.waitKey(0)
cv2.destroyAllWindows()