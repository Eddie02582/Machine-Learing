# -*- coding: utf-8 -*-
import sys
import dlib
import cv2

tracker = dlib.correlation_tracker() # 導入correlation_tracker()類
cap = cv2.VideoCapture(0) # OpenCV打開攝像頭
start_flag = True # 標記，是否是第一幀，若在第一幀需要先初始化
selection = None # 實時跟踪鼠標的跟踪區域
track_window = None # 要檢測的物體所在區域
drag_start = None # 標記，是否開始拖動鼠標

# 鼠標點擊事件回調函數
def onMouseClicked(event, x, y, flags, param):
    global selection, track_window, drag_start # 定義全局變量
    if event == cv2.EVENT_LBUTTONDOWN: # 鼠標左鍵按下
        drag_start = (x, y)
        track_window = None
    if drag_start: # 是否開始拖動鼠標，記錄鼠標位置
        xMin = min(x, drag_start[0])
        yMin = min(y, drag_start[1])
        xMax = max(x, drag_start[0])
        yMax = max(y, drag_start[1])
        selection = (xMin, yMin, xMax, yMax)
    if event == cv2.EVENT_LBUTTONUP: # 鼠標左鍵鬆開
        drag_start = None
        track_window = selection
        selection = None

if __name__ == '__main__':
    cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("image", onMouseClicked)

    # opencv的bgr格式圖片轉換成rgb格式
    # b, g, r = cv2.split(frame)
    # frame2 = cv2.merge([r, g, b])

    while(1):
        ret, frame = cap.read() # 從攝像頭讀入1幀

        if start_flag == True: # 如果是第一幀，需要先初始化
            # 這裡是初始化，窗口中會停在當前幀，用鼠標拖拽一個框來指定區域，隨後會跟踪這個目標；我們需要先找到目標才能跟踪不是嗎？
            while True:
                img_first = frame.copy() # 不改變原來的幀，拷貝一個新的出來
                if track_window: # 跟踪目標的窗口畫出來了，就實時標出來
                    cv2.rectangle(img_first, (track_window[0], track_window[1]), (track_window[2], track_window[3]), (0,0,255), 1)
                elif selection: # 跟踪目標的窗口隨鼠標拖動實時顯示
                    cv2.rectangle(img_first, (selection[0], selection[1]), (selection[2], selection[3]), (0,0,255), 1)
                cv2.imshow("image", img_first)
                # 按下回車，退出循環
                if cv2.waitKey(5) == 13:
                    break
            start_flag = False # 初始化完畢，不再是第一幀了
            tracker.start_track(frame, dlib.rectangle(track_window[0], track_window[1], track_window[2], track_window[3])) # 跟踪目標，目標就是選定目標窗口中的
        else:
            tracker.update(frame) # 更新，實時跟踪

        box_predict = tracker.get_position() # 得到目標的位置
        print (box_predict)
        cv2.rectangle(frame,(int(box_predict.left()),int(box_predict.top())),(int(box_predict.right()),int(box_predict.bottom())),(0,255,255), 1) # 用矩形框標註出來
        cv2.imshow("image", frame)
        # 如果按下ESC鍵，就退出        
        if cv2.waitKey(10) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()