# Face Recognition with camera


這邊把他寫成clss 方便呼叫


```python
import dlib,os,sys,glob
import numpy as np
from skimage import io
import cv2
import imutils

class FaceRecognition():
    def __init__(self):
        self.shape_predictor_path = "../dat/shape_predictor_68_face_landmarks.dat"
        self.face_recognition_model_path = "../dat/dlib_face_recognition_resnet_model_v1.dat"
        self.face_npy_path = "./resources"

        self.detector = dlib.get_frontal_face_detector()
    
        #人臉68特徵點模型檢測器
        self.shape_predictor = dlib.shape_predictor(self.shape_predictor_path) 

        #載入人臉辨識模型及檢測器
        self.face_rec_model = dlib.face_recognition_model_v1(self.face_recognition_model_path) 
        self.candidate_data = []

    def get_face_npy(self):
        npy_files = glob.glob(os.path.join(self.face_npy_path,"*.npy"))
        return npy_files
    
    def load_person_data(self):
        npy_files = self.get_face_npy()        
        for npy_file in npy_files:
            base = os.path.basename(npy_file)
            name = os.path.splitext(base)[0]
            vectors = np.load(npy_file)
            self.candidate_data.append([name,vectors])    
        
    def get_image_description(self,img): 
        #img = io.read(file)            
        dets = self.detector(img,1)
        # 取出所有偵測的結果
        descriptors = []
        for k, d in enumerate(dets):
            #68特徵點偵測
            shape = self.shape_predictor(img,d)
                
            #128維特徵向量描述
            face_descriptor = self.face_rec_model.compute_face_descriptor(img,shape)
                
            #轉換numpy array 格式
            v = np.array(face_descriptor) 
            #descriptors.append(v)
            return v
        return []
        
    def get_face_recognition(self,img):
        face_descriptor =  self.get_image_description(img)        
        
        candidate_distance = [] 
        name = ""  
       
        if face_descriptor != []:
            for name,vectors in self.candidate_data:
                dist_ = np.linalg.norm(vectors - face_descriptor)
                candidate_distance.append([name,dist_])   

            sorted(candidate_distance,key = lambda x:x[1])            
            print (candidate_distance)
            name,score = candidate_distance[0]            
            if score > 0.6:
                name = "vistor"        
        return name         

```

```python
import dlib,os,sys,glob
import numpy as np
from skimage import io
import numpy as np
import cv2
import imutils
import face_rec as fc



def main():
    cap = cv2.VideoCapture(0)    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    width,height = 640,480
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (width, height))    
    
    face_rec = fc.FaceRecognition()
    face_rec.load_person_data()
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        
        name = face_rec.get_face_recognition(frame)
        if name:
            cv2.putText(frame, name, (15, 15), cv2.FONT_HERSHEY_DUPLEX,0.7, (255, 255, 0), 1, cv2.LINE_AA)        
        out.write(frame)
        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break;
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
```