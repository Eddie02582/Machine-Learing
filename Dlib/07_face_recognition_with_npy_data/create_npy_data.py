import dlib,os,sys,glob
import numpy
from skimage import io
import numpy as np
import cv2
import imutils



def main():
    
    if len(sys.argv) != 2:
        exit()

    face_npy_path = "./resources"    
    img_name = sys.argv[1]
   
    # Dlib 的人臉偵測器
    detector = dlib.get_frontal_face_detector()
    
    #人臉68特徵點模型檢測器
    shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 

    #載入人臉辨識模型及檢測器
    face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")   

    
    img = io.imread(img_name)
    name = os.path.splitext(img_name)[0]
    filePath = name + '.npy'
    
    
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


main()

