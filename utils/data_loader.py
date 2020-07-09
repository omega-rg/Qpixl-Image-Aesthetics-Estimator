# Image data generator modules

import numpy as np
import os
import glob

import tensorflow as tf

# directory where images are stored
base_images_path = "D:\\Real Estate CV\\AVA_dataset\\images\\images\\"
# text file containing the image id's and their scores (AVA.txt in the AVA dataset)
ava_dataset_path = "D:\\Real Estate CV\\AVA_dataset\\AVA.txt"

# image dimensions
img_size=224

# image paths
train_img_paths = []
# image scores
train_scores = []

file_handler=open(ava_dataset_path)
lines=file_handler.readlines()

# storing paths of training images and their scores in two numpy arrays
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
        train_scores.append(values)
    
    parts=255000//20
    if (i%parts==0) and i!=0:
        print("Loaded %d%% of the dataset.. %d.."%((i/parts)*5,i))

train_img_paths=np.array(train_img_paths)
train_scores=np.array(train_scores)


#Train-Val Split
train_img_paths=train_img_paths[:-5000]
train_scores=train_scores[:-5000]
val_img_paths=train_img_paths[-5000:]
val_scores=train_scores[-5000:]

print("Train Set Size = %d" % train_img_paths.shape[0])
print("Validation Set Size = %d" % val_img_paths.shape[0])


# preprocessing image without augmentation
def parse_data_without_augmentation(file_path,scores):
    img=tf.io.read_file(file_path) 
    img=tf.io.decode_jpeg(img,channels=3) #img-type=uint8, range=(0,255)
    img=tf.image.resize(img,(img_size,img_size)) #img-type=float32, range=(0,255)
    img=(img-127.5)/127.5
    return img,scores

# preprocessing image with augmentation
def parse_data_with_augmentation(file_path,scores):
    img=tf.io.read_file(file_path) 
    img=tf.io.decode_jpeg(img,channels=3) #img-type=uint8, range=(0,255)
    img=tf.image.resize(img,(256,256))
    img=tf.image.random_crop(img,[img_size,img_size,3])
    img=tf.image.random_flip_left_right(img)
    img=(img-127.5)/127.5
    return img,scores

# generator for training images
def train_generator(batchsize,shuffle=True):
    dataset=tf.data.Dataset.from_tensor_slices((train_img_paths,train_scores))
    dataset=dataset.map(parse_data_with_augmentation)
    dataset=dataset.batch(batchsize)
    dataset=dataset.repeat()
    if (shuffle):
        dataset=dataset.shuffle(5)
    for i,batch in enumerate(dataset):
        try:
            yield batch
        except:
            continue

# generator for validation images
def val_generator(batchsize,shuffle=True):
    dataset=tf.data.Dataset.from_tensor_slices((train_img_paths,train_scores))
    dataset=dataset.map(parse_data_without_augmentation,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset=dataset.batch(batchsize)
    dataset=dataset.repeat()
    if (shuffle):
        dataset=dataset.shuffle(5)
    for i,batch in enumerate(dataset):
        try:
            yield batch
        except:
            continue




