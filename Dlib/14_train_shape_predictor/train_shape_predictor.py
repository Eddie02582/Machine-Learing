import os
import sys
import glob

import dlib



faces_folder = './faces'

options = dlib.shape_predictor_training_options()

#設定參數
options.oversampling_amount = 300
options.nu = 0.05
options.tree_depth = 2
options.be_verbose = True


training_xml_path = os.path.join(faces_folder, "training_with_face_landmarks.xml")
#進行training
dlib.train_shape_predictor(training_xml_path, "predictor.dat", options)

#traing數據準確度
print("\nTraining accuracy: {}".format(dlib.test_shape_predictor(training_xml_path, "predictor.dat")))

# test數據準確度
testing_xml_path = os.path.join(faces_folder, "testing_with_face_landmarks.xml")
print("Testing accuracy: {}".format(dlib.test_shape_predictor(testing_xml_path, "predictor.dat")))

