# Training Model


## imglab
Dlib�����Ѥ@�� tool�s imglab�i�H���U�s�@�V�m�Ϊ��ƾ�,�w�ˤ覡�p�U
�ݥ��w��cmake ...

```
    cd dlib/tools/imglab
    mkdir build
    cd build
    cmake ..
    cmake --build . --config Release
```
�w�˦��\��A�i�H�� dlib/tools/imglab/build����� imglab�ɮ�

���� imglab -c xml_file img_folder

```
    imglab -c train_data.xml ./images
```
�����|����train_data.xml,�̭����e�]�timages�Ҧ��Ϥ���T

```xml
<?xml version='1.0' encoding='ISO-8859-1'?>
<?xml-stylesheet type='text/xsl' href='image_metadata_stylesheet.xsl'?>
<dataset>
<name>imglab dataset</name>
<comment>Created by imglab tool.</comment>
<images>
  <image file='images\image_0001.jpg'>
  </image>
  <image file='images\image_0002.jpg'>
  </image>
  <image file='images\image_0003.jpg'>
  </image>
  <image file='images\image_0004.jpg'>
  </image>
 
  ...
</images>
 ```
  
����
```
    imglab train_data.xml
```
<img src="imglab-1.PNG">

���� Shift + �ƹ��������n�����������C����ηƹ�������������خءA�خ��ܫC���delete�R��
<img src="imglab-2.PNG">

�b Menu/File���I�� Save�A�N�i�H���誺�Ъ`�x�s�b mydataset.xml���C
  
  
  
�p�G�ݭn��@�ǯS�x�i��Ъ`�C����ۭ誺 mydataset.xml�~��B�z�C���]�ȹ�Ϲ��е����ӯS�x�C
./imglab train_data.xml --parts "1 2 3 4 5"
<img src="imglab-3.PNG">  
  
  
  
�i�H�o�{train_data.xml �P�쥻�h�Fbox tag
  
```xml
<?xml version='1.0' encoding='ISO-8859-1'?>
<?xml-stylesheet type='text/xsl' href='image_metadata_stylesheet.xsl'?>
<dataset>
<name>imglab dataset</name>
<comment>Created by imglab tool.</comment>
<images>
  <image file='images\image_0001.jpg'>
    <box top='46' left='35' width='261' height='171'/>
  </image>
  <image file='images\image_0002.jpg'>
    <box top='35' left='45' width='184' height='128'/>
  </image>
  <image file='images\image_0003.jpg'>
    <box top='58' left='38' width='241' height='167'/>
  </image>
  <image file='images\image_0004.jpg'>
    <box top='42' left='36' width='241' height='169'/>
  </image>
  <image file='images\image_0005.jpg'>
    <box top='12' left='6' width='285' height='163'/>
  </image>
  <image file='images\image_0006.jpg'>
    <box top='14' left='14' width='255' height='175'/>
  </image>
  <image file='images\image_0007.jpg'>
    <box top='64' left='38' width='183' height='205'/>
  </image>
  ...
```  
  
## traning data
�o��Ntestdata���յ��� 
```python

# -*- coding: utf-8 -*-
import os
import sys
import glob
import dlib
import cv2

# options�Ω�]�m�V�m���ѼƩM�Ҧ�
options = dlib.simple_object_detector_training_options()
# Since faces are left/right symmetric we can tell the trainer to train a
# symmetric detector.  This helps it get the most value out of the training
# data.

options.add_left_right_image_flips = True


# ����V�q����C�ѼơA�q�`�q�{����5.�ۤv�A����ѼƥH�F��̦n���ĪG
options.C = 5
# �u�{�ơA�A�q����4�֪��ܴN��4
options.num_threads = 4
options.be_verbose = True


current_path = os.getcwd()
train_folder = current_path + '/elephant_train/'
#test_folder = current_path + '/elephant_test/'
train_xml_path = train_folder + 'train_data.xml'
#test_xml_path = test_folder + 'elephant.xml'

print("training file path:" + train_xml_path)
# print(train_xml_path)
#print("testing file path:" + test_xml_path)
# print(test_xml_path)


print("start training:")
dlib.train_simple_object_detector(train_xml_path, 'detector.svm', options)

print("")  # Print blank line to create gap from previous output
print("Training accuracy: {}".format(
    dlib.test_simple_object_detector(train_xml_path, "detector.svm")))

# print("Testing accuracy: {}".format(
    # dlib.test_simple_object_detector(test_xml_path, "detector.svm")))
 

```
## test data
```python

import os
import sys
import dlib
import cv2
import glob

detector = dlib.simple_object_detector("detector.svm")

current_path = os.getcwd()
test_folder = current_path + '/elephant_test/images/'

print (test_folder)

for f in glob.glob(test_folder+'*.jpg'):
    print("Processing file: {}".format(f))
    img = cv2.imread(f, cv2.IMREAD_COLOR)
    b, g, r = cv2.split(img)
    img2 = cv2.merge([r, g, b])
    dets = detector(img2)
    print("Number of faces detected: {}".format(len(dets)))
    for index, face in enumerate(dets):
        print('face {}; left {}; top {}; right {}; bottom {}'.format(index, face.left(), face.top(), face.right(), face.bottom()))

        left = face.left()
        top = face.top()
        right = face.right()
        bottom = face.bottom()
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 3)
    cv2.namedWindow(f, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(f, img)

k = cv2.waitKey(0)
cv2.destroyAllWindows()













```
  