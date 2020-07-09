# Python libraries
import numpy as np
import argparse
from pathlib import Path
import os
import re
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Tensorflow libraries
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.preprocessing.image import load_img,img_to_array

# Utility files
from utils.nasnet import NASNetMobile, preprocess_input
from utils.score_utils import mean_score,std_score

# Command line Argument Parser
parser=argparse.ArgumentParser("NIMA evaluation using NasnetMobile")
parser.add_argument("--dirpath",default=None,help="Directory containing the images",type=str)
parser.add_argument("--imgpaths",default=None,help="space_separated_images_path",type=str,nargs='+')
parser.add_argument("--rank",default="true",help="Should images be ranked",type=str)

args=parser.parse_args()
if (args.rank.lower() in ['y','t','1','yes','true']):
    rank=True
else:
    rank=False

# Directory path (if directory was passed as argument)
dirpath=args.dirpath
# Image paths (if space separated image paths were passed)
imgpaths=args.imgpaths

img_path_list=[]
if (dirpath is not None):
    if (dirpath[-2:]=="\\"):
        dirpath=dirpath[:-2]
    img_path_list=[dirpath+"\\"+img for img in os.listdir(dirpath) if re.match(r'.*\.["png","jpg","jpeg"]',img)]
elif (imgpaths is not None):
    img_path_list=[img for img in imgpaths if (re.match(r'.*\.["png","jpg","jpeg"]',img) and os.path.exists(img))]
else:
    raise RuntimeError("Error: Either --dirpath or --imgpaths argument must be provided")

if (len(img_path_list)==0): 
    raise RuntimeError("Error: No images found")

# image dimension
image_size=224

# model declaration
base_model = NASNetMobile(input_shape=(image_size, image_size, 3), weights="imagenet", include_top=False, pooling='avg')
x=Dropout(0.75)(base_model.output)
x=Dense(units=10,activation="softmax")(x)
model=Model(inputs=base_model.input,outputs=x)

# Loading trained model weigths
model.load_weights("weights\\nasnet_weights.h5")

# Path to the result text file
record_file_path=".\\record.txt"
recorder=open(record_file_path,'w')

# Storing predictions in the text file
for img_path in img_path_list:
    img=load_img(img_path,target_size=(image_size,image_size))
    img=img_to_array(img)
    img=np.expand_dims(img,axis=0)
    img=preprocess_input(img)

    scores=model.predict(img,batch_size=1)
    recorder.write(str(img_path)+"\n")
    # print(img_path)
    res=scores[0]
    # print(res)
    recorder.write(str(mean_score(res))+" ")
    recorder.write(str(std_score(res))+" ")
    recorder.write("\n\n")
    # print()

recorder.close()




    







