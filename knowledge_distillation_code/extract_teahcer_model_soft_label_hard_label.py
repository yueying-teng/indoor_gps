
import numpy as np
import os
from tqdm import tqdm_notebook
import glob

import keras 
from keras.models import load_model, Model
from keras.layers import Lambda, Activation
from keras.preprocessing import image



# EPOCHS = 50
# BATCH_SIZE = 16
NUM_CLASSES = 76

### load data for teahcer model to extract soft label and hard label

train_data_dir = '/home/ubuntu/.jupyter/indoor_data/train' 
test_data_dir = '/home/ubuntu/.jupyter/indoor_data/test'


def preprocess_input(x):
    '''
    channel mean array([118.82756496,  99.98454557,  85.08010847])
    '''

    x[:, :, 0] -= 118.82756496
    x[:, :, 1] -= 99.98454557
    x[:, :, 2] -= 85.08010847
    
    return x


def get_hard_label_and_feature(data_dir, IMG_SIZE):
    '''
    feature loaded here is for teacher model soft label prediction
    
    '''
    class_dir = glob.glob(os.path.join(data_dir, 'class_*'))

    hard_label = []
    img_list = []

    for i in tqdm_notebook(range(len(class_dir))):
        img_path_list = glob.glob(os.path.join(class_dir[i], 'frame_*.jpg'))
        hard_label.append([int(class_dir[i][-4: ]) for k in range(len(img_path_list))])

        for j in range(len(img_path_list)):
            img = image.load_img(img_path_list[j], target_size = (IMG_SIZE, IMG_SIZE))
            # convert PIL.Image.Image type to 3D tensor 
            x = image.img_to_array(img)
            # zero center channel mean
            x = preprocess_input(x)
            img_list.append(x)
            
#         if i != 0:
#             break
            
    hard_label = [k for sub in hard_label for k in sub]
    # label starts from 1, change it to 0 here
    hard_label = np.array(hard_label) - 1

    # one hot encode the hard labels 
    hard_label = keras.utils.to_categorical(hard_label, NUM_CLASSES)

    arr = np.array(img_list)
        
    return hard_label, arr


hard_label_train, x_train = get_hard_label_and_feature(train_data_dir, IMG_SIZE = 32)
# print(len(hard_label_train))
print(x_train.shape)

hard_label_test, x_test = get_hard_label_and_feature(test_data_dir, IMG_SIZE = 32)
# print(len(hard_label_test))
print(x_test.shape)


### extract teacher model soft label 

soft_temperature = 5

teacher_model = load_model('./models/2019-01-30_indoor_gps_RAN_model_0.0001.h5')
teacher_model.layers.pop()

logits = teacher_model.layers[-1].output
logits_soft = Lambda(lambda x: x/ soft_temperature, name = 'high_temp')(logits)
prob =  Activation('softmax', name = 'soft_activation')(logits_soft)


teacher_model_soft = Model(teacher_model.input, prob)


# soft label from teacher_model on both training and testing data
# to be used as true labels in training and testing the distilled model

soft_label_train = teacher_model_soft.predict(x_train)
soft_label_test = teacher_model_soft.predict(x_test)

print (soft_label_train.shape, soft_label_test.shape)


# truth for training and validation (split = 0.001)
new_train_label = np.hstack((hard_label_train, soft_label_train))

# truth for testing
new_test_label = np.hstack((hard_label_test, soft_label_test))

print (new_train_label.shape, new_test_label.shape)


np.save('new_train_label.npy', new_train_label)
np.save('new_test_label.npy', new_test_label)

