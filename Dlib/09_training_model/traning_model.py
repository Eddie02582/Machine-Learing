# -*- coding: utf-8 -*-
import os
import sys
import glob
import dlib
import cv2

# options用於設置訓練的參數和模式
options = dlib.simple_object_detector_training_options()
# Since faces are left/right symmetric we can tell the trainer to train a
# symmetric detector.  This helps it get the most value out of the training
# data.

options.add_left_right_image_flips = True


# 支持向量機的C參數，通常默認取為5.自己適當更改參數以達到最好的效果
options.C = 5
# 線程數，你電腦有4核的話就填4
options.num_threads = 4
options.be_verbose = True

# 获取路径
current_path = os.getcwd()
train_folder = current_path + '/elephant_train/'
#test_folder = current_path + '/cats_test/'
train_xml_path = train_folder + 'train_data.xml'
#test_xml_path = test_folder + 'cats.xml'

print("training file path:" + train_xml_path)
# print(train_xml_path)
#print("testing file path:" + test_xml_path)
# print(test_xml_path)

# 开始训练
print("start training:")
dlib.train_simple_object_detector(train_xml_path, 'detector.svm', options)

print("")  # Print blank line to create gap from previous output
print("Training accuracy: {}".format(
    dlib.test_simple_object_detector(train_xml_path, "detector.svm")))

# print("Testing accuracy: {}".format(
    # dlib.test_simple_object_detector(test_xml_path, "detector.svm")))
    
    
    
    
    
    