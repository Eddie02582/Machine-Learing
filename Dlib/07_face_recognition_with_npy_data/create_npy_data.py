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
    # Dlib 的人臉偵測器
    detector = dlib.get_frontal_face_detector()
    
    #人臉68特徵點模型檢測器
    shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 

    #載入人臉辨識模型及檢測器
    face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")   

    #抓取face_data_path底下副檔名為.jpg
    img_files = glob.glob(os.path.join(face_data_path,"*.jpg"))
    for img_file in img_files:    
        img = io.imread(img_file)     
        
        #取出檔名
        name = Path(img_file).stem    
        filePath = face_npy_path + name + '.npy'             
    
        face_rects = detector(img,1)
        
        #這邊只取辨識第一位
        for k, d in enumerate(face_rects):  
            #68特徵點偵測
            shape = shape_predictor(img,d)
            #128維特徵向量描述
            face_descriptor = face_rec_model.compute_face_descriptor(img,shape)
            #轉換numpy array 格式
            face_descriptor = np.array(face_descriptor)
            break
    
    
        np.save(filePath, face_descriptor)

create_npy_data()


