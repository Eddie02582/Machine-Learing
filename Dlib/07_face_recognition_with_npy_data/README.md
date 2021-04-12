# Face Recognition with npy data


�e�@�g���Цp��q��Ƨ�Ū���ɮ׿���,�o�g���N���ɸ����s��.npy ��,���ѮɦbŪ��<br>


## ����npy data


```python
import dlib,os,glob
import numpy
from skimage import io
import numpy as np
import cv2
import imutils
from pathlib import Path

def create_npy_data():

    face_npy_path = "./resources/"      
    face_data_path = "./face"
    # Dlib ���H�y������
    detector = dlib.get_frontal_face_detector()
    
    #�H�y68�S�x�I�ҫ��˴���
    shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 

    #���J�H�y���Ѽҫ����˴���
    face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")   

    
    img_files = glob.glob(os.path.join(face_data_path,"*.jpg"))
    for img_file in img_files:    
        img = io.imread(img_file)     
  
        name = Path(img_file).stem    
        filePath = face_npy_path + name + '.npy'             
    
        face_rects = detector(img,1)
        
        #�o��u�����ѲĤ@��
        for k, d in enumerate(face_rects):  
            #68�S�x�I����
            shape = shape_predictor(img,d)
            #128���S�x�V�q�y�z
            face_descriptor = face_rec_model.compute_face_descriptor(img,shape)
            #�ഫnumpy array �榡
            face_descriptor = np.array(face_descriptor)
            break
    
    
        np.save(filePath, face_descriptor)
```

## ��e�@�g�@��,�t�b�ϥ�np.loadŪ��.npy�ɪ�data

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
    face_npy_path = "./resources"    
    img_name = sys.argv[1]
   
    # Dlib ���H�y������
    detector = dlib.get_frontal_face_detector()
    
    #�H�y68�S�x�I�ҫ��˴���
    shape_predictor = dlib.shape_predictor("../dat/shape_predictor_68_face_landmarks.dat") 

    #���J�H�y���Ѽҫ����˴���
    face_rec_model = dlib.face_recognition_model_v1("../dat/dlib_face_recognition_resnet_model_v1.dat")     

    candidate_data = []

    npy_files = glob.glob(os.path.join(face_npy_path,"*.npy"))

    #���o�����W�r���V�q
    for npy_file in npy_files:
        base = os.path.basename(npy_file)
        name = os.path.splitext(base)[0]
        vectors = np.load(npy_file)
        candidate_data.append([name,vectors])
        
            
    #���ѹϤ��B�z
    img = io.imread(img_name)
    face_rects = detector(img,1)
    distance = []
    
    #�o��u�����ѲĤ@��
    for k, d in enumerate(face_rects):  
        shape = shape_predictor(img,d)
        face_descriptor = face_rec_model.compute_face_descriptor(img,shape)
        d_test = np.array(face_descriptor)
        x1 = d.left()
        y1 = d.top()
        x2 = d.right()
        y2 = d.bottom()         
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)
        break
     
    candidate_distance_dict = {}
    
    #�p����ѹϤ��P���ɪ��V�q�t��
    for candidate,vectors in  candidate_data:
        dist_ = np.linalg.norm(vectors - d_test)
        candidate_distance_dict[candidate] = dist_     
        
    
    candidate_distance_dict_sorted = sorted(candidate_distance_dict.items(),key = lambda d:d[1])
    
    print (candidate_distance_dict_sorted)
    result = candidate_distance_dict_sorted[0][0]

    cv2.putText(img, result, (x1, y1), cv2.FONT_HERSHEY_DUPLEX,0.7, (255, 255, 255), 1, cv2.LINE_AA)

    #img = imutils.resize(img, width=800)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    cv2.imshow("test",img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
```