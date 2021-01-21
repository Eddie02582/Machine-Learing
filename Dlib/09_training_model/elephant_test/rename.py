import argparse
import glob
import os
import re

index = 0

path ='./images/'
#path = os.path.abspath(path)

image_files = glob.glob(os.path.join(path,"*.jpg"))
for image in image_files:
    dst = "image_{0:04d}.jpg".format(index)
    print (image)
    os.rename(image, path + dst)
    index += 1