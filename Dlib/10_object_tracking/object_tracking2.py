# -*- coding: utf-8 -*-
import sys
import dlib
import cv2

class myCorrelationTracker(object):
    def __init__(self, windowName='default window', cameraNum=0):
        # 自定義幾個狀態標誌
        self.STATUS_RUN_WITHOUT_TRACKER = 0 # 不跟踪目標，但是實時顯示
        self.STATUS_RUN_WITH_TRACKER = 1 # 跟踪目標，實時顯示
        self.STATUS_PAUSE = 2 # 暫停，卡在當前幀
        self.STATUS_BREAK = 3 # 退出
        self.status = self.STATUS_RUN_WITHOUT_TRACKER # 指示狀態的變量

        # 這幾個跟前面程序1定義的變量一樣
        self.track_window = None # 實時跟踪鼠標的跟踪區域
        self.drag_start = None # 要檢測的物體所在區域
        self.start_flag = True # 標記，是否開始拖動鼠標

        # 創建好顯示窗口
        cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(windowName, self.onMouseClicked)
        self.windowName = windowName

        # 打開攝像頭
        self.cap = cv2.VideoCapture(cameraNum)

        # correlation_tracker()類，跟踪器，跟程序1中一樣
        self.tracker = dlib.correlation_tracker()

        # 當前幀
        self.frame = None

    # 按鍵處理函數
    def keyEventHandler(self):
        keyValue = cv2.waitKey(5) # 每隔5ms讀取一次按鍵的鍵值
        if keyValue == 27: # ESC
            self.status = self.STATUS_BREAK
        if keyValue == 32: # 空格
            if self.status != self.STATUS_PAUSE: # 按下空格，暫停播放，可以選定跟踪的區域
                #print self.status
                self.status = self.STATUS_PAUSE
                #print self.status
            else: # 再按次空格，重新播放，但是不進行目標識別
                if self.track_window:
                    self.status = self.STATUS_RUN_WITH_TRACKER
                    self.start_flag = True
                else:
                    self.status = self.STATUS_RUN_WITHOUT_TRACKER
        if keyValue == 13: # 回車
            #print '**'
            if self.status == self.STATUS_PAUSE: # 按下空格之後
                if self.track_window: # 如果選定了區域，再按回車，表示確定選定區域為跟踪目標
                    self.status = self.STATUS_RUN_WITH_TRACKER
                    self.start_flag = True

    # 任務處理函數
    def processHandler(self):
        # 不跟踪目標，但是實時顯示
        if self.status == self.STATUS_RUN_WITHOUT_TRACKER:
            ret, self.frame = self.cap.read()            
            cv2.imshow(self.windowName, self.frame)
        # 暫停，暫停時使用鼠標拖動紅框，選擇目標區域，與程序1類似
        elif self.status == self.STATUS_PAUSE:
            img_first = self.frame.copy() # 不改變原來的幀，拷貝一個新的變量出來
            if self.track_window: # 跟踪目標的窗口畫出來了，就實時標出來
                cv2.rectangle(img_first, (self.track_window[0], self.track_window[1]), (self.track_window[2], self.track_window[3]), (0,0,255), 1)
            elif self.selection: # 跟踪目標的窗口隨鼠標拖動實時顯示
                cv2.rectangle(img_first, (self.selection[0], self.selection[1]), (self.selection[2], self.selection[3]), (0,0,255), 1)
            cv2.imshow(self.windowName, img_first)
        # 退出
        elif self.status == self.STATUS_BREAK:
            self.cap.release() # 釋放攝像頭
            cv2.destroyAllWindows() # 釋放窗口
            sys.exit() # 退出程序
        # 跟踪目標，實時顯示
        elif self.status == self.STATUS_RUN_WITH_TRACKER:
            ret, self.frame = self.cap.read() # 從攝像頭讀取一幀
            if self.start_flag: # 如果是第一幀，需要先初始化
                self.tracker.start_track(self.frame, dlib.rectangle(self.track_window[0], self.track_window[1], self.track_window[2], self.track_window[3])) # 開始跟踪目標
                self.start_flag = False # 不再是第一幀
            else:
                self.tracker.update(self.frame) # 更新

                # 得到目標的位置，並顯示
                box_predict = self.tracker.get_position()
                if box_predict:
                    print (box_predict)
                    cv2.rectangle(self.frame,(int(box_predict.left()),int(box_predict.top())),(int(box_predict.right()),int(box_predict.bottom())),(0,255,255 ),1)
                    cv2.imshow(self.windowName, self.frame)

    # 鼠標點擊事件回調函數
    def onMouseClicked(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN: # 鼠標左鍵按下
            self.drag_start = (x, y)
            self.track_window = None
        if self.drag_start: # 是否開始拖動鼠標，記錄鼠標位置
            xMin = min(x, self.drag_start[0])
            yMin = min(y, self.drag_start[1])
            xMax = max(x, self.drag_start[0])
            yMax = max(y, self.drag_start[1])
            self.selection = (xMin, yMin, xMax, yMax)
        if event == cv2.EVENT_LBUTTONUP: # 鼠標左鍵鬆開
            self.drag_start = None
            self.track_window = self.selection
            self.selection = None

    def run(self):
        while(1):
            self.keyEventHandler()
            self.processHandler()


if __name__ == '__main__':
    testTracker = myCorrelationTracker(windowName='image', cameraNum=0)
    testTracker.run()