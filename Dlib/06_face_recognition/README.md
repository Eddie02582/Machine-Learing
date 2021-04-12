# Recognition Face Image

```python
import dlib,os,sys,glob
import numpy
from skimage import io
import numpy as np
import cv2
import imutils
import os
import sys


def main():
    
    if len(sys.argv) != 2:
        exit()
    
    os.chdir(sys.path[0])   
    #欲辨識圖片
    img_name = sys.argv[1]    
    face_data_path = "./face"    
    
    # Dlib 的人臉偵測器
    detector = dlib.get_frontal_face_detector()
    
    #人臉68特徵點模型檢測器
    shape_predictor = dlib.shape_predictor("../dat/shape_predictor_68_face_landmarks.dat") 

    #載入人臉辨識模型及檢測器
    face_rec_model = dlib.face_recognition_model_v1("../dat/dlib_face_recognition_resnet_model_v1.dat")   

    descriptors  = []

    candidate = []

    img_files = glob.glob(os.path.join(face_data_path,"*.jpg"))

    for file in img_files:
        base = os.path.basename(file)
        candidate.append(os.path.splitext(base)[0])
        img = io.imread(file)
        
        dets = detector(img,1)
        
        # 取出所有偵測的結果
        for k, d in enumerate(dets):
            #68特徵點偵測
            shape = shape_predictor(img,d)
            
            #128維特徵向量描述
            face_descriptor = face_rec_model.compute_face_descriptor(img,shape)
            
            #轉換numpy array 格式
            v = np.array(face_descriptor)
            descriptors.append(v)
            
    #辨識圖片處理
    img = io.imread(img_name)
    face_rects = detector(img,1)
    distance = []
    for k, d in enumerate(face_rects):
        #68特徵點偵測
        shape = shape_predictor(img,d)
        
        #128維特徵向量描述
        face_descriptor = face_rec_model.compute_face_descriptor(img,shape)
        
        d_test = np.array(face_descriptor)
        x1 = d.left()
        y1 = d.top()
        x2 = d.right()
        y2 = d.bottom()         
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)
        for i in  descriptors:
            #計算向量差值
            dist_ = np.linalg.norm(i - d_test)
            distance.append(dist_)

    
    candidate_distance_dict = dict(zip(candidate,distance))
    
    #依照距離排序
    candidate_distance_dict_sorted = sorted(candidate_distance_dict.items(),key = lambda d:d[1])
    
    
    print (candidate_distance_dict_sorted)
    #取出最相像的
    result = candidate_distance_dict_sorted[0][0]

    cv2.putText(img, result, (x1, y1), cv2.FONT_HERSHEY_DUPLEX,0.7, (255, 255, 255), 1, cv2.LINE_AA)

    #img = imutils.resize(img, width=800)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    cv2.imshow("test",img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



main()



```
