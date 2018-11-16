import pandas as pd
import numpy as np
import os
import glob 
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import datetime

from keras.preprocessing import image
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler


# get image directories in 5 classes with full frames
all_img = glob.glob('/indoor_gps_5cls_all_frame/class*/output/frame*.jpg')

# get image directories in all three class
# all_img = glob.glob('/indoor_gps/*/m*/*.jpg')
 
NUM_CLASSES = 5
# NUM_CLASSES = 3
IMG_SIZE  = 224 
BATCH_SIZE = 16
EPOCHS = 30
img_shape = (IMG_SIZE, IMG_SIZE, 3)




def path_to_tensor(img_path):
    """
    read image data to the four dimensional tensor format required by keras 
    """
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size = (IMG_SIZE, IMG_SIZE))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # x = (x - np.mean(x))/ np.std(x)x
    x -= np.mean(x)
    x /= np.std(x)

    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis = 0)


def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    
    return np.vstack(list_of_tensors)


# image to tensor
img_array = paths_to_tensor(all_img).astype('float32')/255
# img_array = paths_to_tensor(all_img).astype(np.uint8)/255

# label to one hot encoded
label_list = []
for i in range(len(all_img)):
    img_dir = all_img[i]
    start = img_dir.find('class')
    end = img_dir.find('/', start)

    label_list.append(img_dir[start+ 6: end])

label_dummy_list = pd.get_dummies(label_list)
label_array = np.asarray(label_dummy_list)



# train_test_split
train_x, test_x, train_y, test_y = train_test_split(img_array, label_array, test_size = 0.1, random_state = 2020)




from keras import regularizers
from keras.optimizers import Adam
from keras.engine.topology import Input
from keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Dense, Flatten, GlobalMaxPooling2D, Lambda, MaxPooling2D, Reshape
from keras.models import Model
from keras import backend as K


def subblock(x, filter, **kwargs):
    x = BatchNormalization()(x)
    y = x
    y = Conv2D(filter, (1, 1), activation='relu', **kwargs)(y) # Reduce the number of features to 'filter'
    y = BatchNormalization()(y)
    y = Conv2D(filter, (3, 3), activation='relu', **kwargs)(y) # Extend the feature field
    y = BatchNormalization()(y)
    y = Conv2D(K.int_shape(x)[-1], (1, 1), **kwargs)(y) # no activation # Restore the number of original features
    y = Add()([x,y]) # Add the bypass connection
    y = Activation('relu')(y)
    return y


def build_model(lr, l2, activation='softmax'):

    regul  = regularizers.l2(l2)
    optim  = Adam(lr = lr, decay = 1e-6)
    kwargs = {'padding':'same', 'kernel_regularizer':regul}


    inp = Input(shape = img_shape) # 384x384x1
    x   = Conv2D(64, (9,9), strides=2, activation='relu', **kwargs)(inp)

    x   = MaxPooling2D((2, 2), strides=(2, 2))(x) # 96x96x64
    for _ in range(2):
        x = BatchNormalization()(x)
        x = Conv2D(64, (3,3), activation='relu', **kwargs)(x)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x) # 48x48x64
    x = BatchNormalization()(x)
    x = Conv2D(128, (1,1), activation='relu', **kwargs)(x) # 48x48x128
    for _ in range(4): x = subblock(x, 64, **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x) # 24x24x128
    x = BatchNormalization()(x)
    x = Conv2D(256, (1,1), activation='relu', **kwargs)(x) # 24x24x256
    for _ in range(4): x = subblock(x, 64, **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x) # 12x12x256
    x = GlobalMaxPooling2D()(x) # 512
    x = Dense(NUM_CLASSES, use_bias = True, activation = activation, name = 'prediction')(x)

    model  = Model(inp, x)
    
    return model


lr = 0.0001
optim  = Adam(lr = lr, decay = 1e-6)

model = build_model(lr = lr, l2 = 0)

# callbacks
# recuding learning rate after each epoch
def lr_schedule(epoch):
    return lr * (0.1 ** int(epoch/ 10))

early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10)
STAMP = "{}_gps_model".format(datetime.date.today().strftime("%Y-%m-%d"))

bst_model_path = "/output/{}.h5".format(STAMP)
model_checkpoint = ModelCheckpoint(bst_model_path,
                                    monitor = 'val_loss',
                                    save_best_only = True,
                                    save_weights_only = True)

model.compile(optimizer = optim, loss = 'categorical_crossentropy', metrics = ["accuracy"])


# training 
history = model.fit(train_x, train_y, batch_size = BATCH_SIZE, epochs = EPOCHS,
                   validation_data = (test_x, test_y),
                   shuffle = True,
                   callbacks = [LearningRateScheduler(lr_schedule), model_checkpoint, early_stopping])


# print validation accuracy from the best model during training 
model.load_weights(bst_model_path)

bst_val_acc = max(history.history['val_acc'])
print("Best val acc: {:.1%}".format(bst_val_acc))

