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



def cnn_model():
    model = Sequential()
    
    # zero padding
    model.add(Conv2D(32, (3, 3), padding = 'same', input_shape = (IMG_SIZE, IMG_SIZE, 3), activation = 'relu'))
    model.add(Conv2D(32, (3, 3), activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (3, 3), padding = 'same', activation = 'relu'))
    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(128, (3, 3), padding = 'same', activation = 'relu'))
    model.add(Conv2D(128, (3, 3), activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation = 'softmax'))
    
    return model



model = cnn_model()


# callbacks 
# lr = 0.01
lr = 0.001

# opt = SGD(lr = lr, decay = 1e-6, momentum = 0.9, nesterov = True)
opt = Adam(lr = lr, decay = 1e-6)

# recuding learning rate after each epoch
def lr_schedule(epoch):
    return lr * (0.1 ** int(epoch/ 10))

# checkpoints
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10)
STAMP = "{}_gps_model".format(datetime.date.today().strftime("%Y-%m-%d"))

bst_model_path = "/output/{}.h5".format(STAMP)
model_checkpoint = ModelCheckpoint(bst_model_path,
                                   save_best_only = True,
                                   save_weights_only = True)

model.compile(optimizer = opt, loss = 'categorical_crossentropy',
              metrics = ["accuracy"])


# training 
history = model.fit(train_x, train_y, batch_size = BATCH_SIZE, epochs = EPOCHS,
                   validation_data = (test_x, test_y),
                   shuffle = True,
                   callbacks = [LearningRateScheduler(lr_schedule), model_checkpoint, early_stopping])


# print validation accuracy from the best model during training 
model.load_weights(bst_model_path)

bst_val_acc = max(history.history['val_acc'])
print("Best val acc: {:.1%}".format(bst_val_acc))

