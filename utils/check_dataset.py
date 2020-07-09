# Check dataset for corrupt image files

import numpy as np
import os
import glob

import tensorflow as tf

base_images_path = "D:\\Real Estate CV\\AVA_dataset\\images\\images\\"
ava_dataset_path = "D:\\Real Estate CV\\AVA_dataset\\AVA.txt"


train_img_paths = []
train_vals = []

file_handler=open(ava_dataset_path)
lines=file_handler.readlines()

for i,file_line in enumerate(lines):
    img=file_line.split()
    img_id=img[1]
    values=np.array(img[2:12],dtype="float32")
    values/=np.sum(values)

    img_path=base_images_path+"%s.jpg"%img_id
    # if (i==0):
    #     print(img_path)

    if os.path.exists(img_path):
        train_img_paths.append(img_path)
        train_vals.append(values)
    
    parts=255000//20
    if (i%parts==0) and i!=0:
        print("Loaded %d%% of the dataset.. %d.."%((i/parts)*5,i))

train_img_paths=np.array(train_img_paths)
train_vals=np.array(train_vals)

def parse(file_path):
    print(file_path)
    path=tf.io.read_file(file_path)
    img=tf.io.decode_image(path,channels=3)
    img=tf.image.convert_image_dtype(img,dtype=tf.float32)
    return img

# print(parse(r"D:\Real Estate CV\AVA_dataset\images\images\1000.jpg"))

cnt=0
for path in train_img_paths:
    # print(path)
    try:
        img=parse(path)
    except Exception as e:
        print("Failed to load: %s"%path)
        cnt+=1

print("Total load fails = %d"%cnt)

