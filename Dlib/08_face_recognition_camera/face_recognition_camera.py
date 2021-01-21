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
        print (name)
        out.write(frame)
        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break;
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    
main()