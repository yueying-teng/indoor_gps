
'''
in order to center the data around zero mean, find the channal mean using all training data 
and use the finding in preprocessing_function in ImageDataGenerator with flow_from_directory 

'''

import numpy as np
import glob
import os
from keras.preprocessing import image
from tqdm import tqdm_notebook


# train_root = './data/train'
train_root = './indoor_data/train'
train_class_list = glob.glob(os.path.join(train_root, 'class_*'))
train_class_list[0]



# find feature mean
channel_sum = []

for i in range(3):
    channel_sum.append(np.mean(one_class_np[:, :, :, i]))
    
print (channel_sum)


class_size = []
channel_sum = [0 for i in range(3)]

for i in tqdm_notebook(range(len(train_class_list))):
    class_image_list = glob.glob(os.path.join(train_class_list[i], '*.jpg'))
    class_size.append(len(class_image_list))
    
    one_class_np_lst = []
    
    for j in range(len(class_image_list)):
        img = image.load_img(class_image_list[i])
        img_np = image.img_to_array(img)
        one_class_np_lst.append(img_np)
    
    one_class_np = np.asarray(one_class_np_lst)
    for k in range(3):
        channel_sum[k] += np.sum(one_class_np[:, :, :, k])


channel_sum
# [2900384272384.0, 2440457342976.0, 2076664691200.0]

np.asarray(channel_sum)/ (sum(class_size)* 1920 *1080)
# array([118.82756496,  99.98454557,  85.08010847])

