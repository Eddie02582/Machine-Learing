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
    #�����ѹϤ�
    img_name = sys.argv[1]    
    face_data_path = "./face"    
    
    # Dlib ���H�y������
    detector = dlib.get_frontal_face_detector()
    
    #�H�y68�S�x�I�ҫ��˴���
    shape_predictor = dlib.shape_predictor("../dat/shape_predictor_68_face_landmarks.dat") 

    #���J�H�y���Ѽҫ����˴���
    face_rec_model = dlib.face_recognition_model_v1("../dat/dlib_face_recognition_resnet_model_v1.dat")   

    descriptors  = []

    candidate = []

    img_files = glob.glob(os.path.join(face_data_path,"*.jpg"))

    for file in img_files:
        base = os.path.basename(file)
        candidate.append(os.path.splitext(base)[0])
        img = io.imread(file)
        
        dets = detector(img,1)
        
        # ���X�Ҧ����������G
        for k, d in enumerate(dets):
            #68�S�x�I����
            shape = shape_predictor(img,d)
            
            #128���S�x�V�q�y�z
            face_descriptor = face_rec_model.compute_face_descriptor(img,shape)
            
            #�ഫnumpy array �榡
            v = np.array(face_descriptor)
            descriptors.append(v)
            
    #���ѹϤ��B�z
    img = io.imread(img_name)
    face_rects = detector(img,1)
    distance = []
    for k, d in enumerate(face_rects):
        #68�S�x�I����
        shape = shape_predictor(img,d)
        
        #128���S�x�V�q�y�z
        face_descriptor = face_rec_model.compute_face_descriptor(img,shape)
        
        d_test = np.array(face_descriptor)
        x1 = d.left()
        y1 = d.top()
        x2 = d.right()
        y2 = d.bottom()         
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)
        for i in  descriptors:
            #�p��V�q�t��
            dist_ = np.linalg.norm(i - d_test)
            distance.append(dist_)

    
    candidate_distance_dict = dict(zip(candidate,distance))
    
    #�̷ӶZ���Ƨ�
    candidate_distance_dict_sorted = sorted(candidate_distance_dict.items(),key = lambda d:d[1])
    
    
    print (candidate_distance_dict_sorted)
    #���X�̬۹���
    result = candidate_distance_dict_sorted[0][0]

    cv2.putText(img, result, (x1, y1), cv2.FONT_HERSHEY_DUPLEX,0.7, (255, 255, 255), 1, cv2.LINE_AA)

    #img = imutils.resize(img, width=800)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    cv2.imshow("test",img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



main()



```
