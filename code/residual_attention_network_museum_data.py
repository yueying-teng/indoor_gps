
from residual_attention_network import *

import numpy as np
import pandas as pd
import os
import datetime

from keras.utils import plot_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD, RMSprop
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau


# model = attention_resnet56()
# plot_model(model, to_file = 'residual_attention_model.png')



def AttentionResNetCifar10(shape=(32, 32, 3), n_channels=32, n_classes=76):
    """
    Attention-56 ResNet for Cifar10 Dataset
    https://arxiv.org/abs/1704.06904
    """
    input_ = Input(shape=shape)
    x = Conv2D(n_channels, (5, 5), padding='same')(input_)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)  # 16x16

    x = residual_block(x, input_channels=32, output_channels=128)
#     x = attention_block_stage2(x, encoder_depth=2)
    x = attention_block_stage2(x)

    x = residual_block(x, input_channels=128, output_channels=256, stride=2)  # 8x8
#     x = attention_block_stage3(x, encoder_depth=1)
    x = attention_block_stage3(x)

    x = residual_block(x, input_channels=256, output_channels=512, stride=2)  # 4x4
#     x = attention_block_stage3(x, encoder_depth=1)
    x = attention_block_stage3(x)

    x = residual_block(x, input_channels=512, output_channels=1024)
    x = residual_block(x, input_channels=1024, output_channels=1024)
    x = residual_block(x, input_channels=1024, output_channels=1024)

    x = AveragePooling2D(pool_size=(4, 4), strides=(1, 1))(x)  # 1x1
    x = Flatten()(x)
    output = Dense(n_classes, activation='softmax')(x)

    model = Model(input_, output)
    
    return model




IMG_SIZE = 32
img_shape = (IMG_SIZE, IMG_SIZE, 3)

train_data_dir = '/home/ubuntu/.jupyter/indoor_data/train' 
validation_data_dir = '/home/ubuntu/.jupyter/indoor_data/test'

# train_data_dir = './data/train' 
# validation_data_dir = './data/test'


EPOCHS = 50
BATCH_SIZE = 16
NUM_CLASSES = 76



# channel mean array([118.82756496,  99.98454557,  85.08010847])
def preprocess_input(x):
    
    x[:, :, 0] -= 118.82756496
    x[:, :, 1] -= 99.98454557
    x[:, :, 2] -= 85.08010847
    
    return x



# train_datagen = ImageDataGenerator(featurewise_center = True,
#                                    featurewise_std_normalization = True)

# test_datagen = ImageDataGenerator(featurewise_center = True,
#                                   featurewise_std_normalization = True)

# # compute quantities required for featurewise normalization
# # (std, mean, and principal components if ZCA whitening is applied)
# train_datagen.fit(train_data)
# test_datagen.fit(train_data)


train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size = (IMG_SIZE, IMG_SIZE),
    batch_size = BATCH_SIZE,
    shuffle = True)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size = (IMG_SIZE, IMG_SIZE),
    batch_size = BATCH_SIZE,
    shuffle = False)




model = AttentionResNetCifar10(n_classes = 76)

lr = 0.0001
# lr = 0.001
lr_reducer = ReduceLROnPlateau(monitor = 'val_acc', factor = 0.2, patience = 7, min_lr = 10e-7, epsilon = 0.01, verbose = 1)
early_stopping = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 8)

STAMP = "{}_indoor_gps_RAN_model_{}".format(datetime.date.today().strftime("%Y-%m-%d"), lr)
bst_model_path = "./models/{}.h5".format(STAMP)

model_checkpoint = ModelCheckpoint(bst_model_path,
                                    monitor = 'val_loss',
                                    save_best_only = True,
                                    save_weights_only = False)


model.compile(optimizer = Adam(lr = lr, decay = 1e-6), loss = 'categorical_crossentropy', metrics = ["accuracy"])


history = model.fit_generator(
    train_generator,
    steps_per_epoch = len(train_generator.classes) // BATCH_SIZE,
    epochs = EPOCHS,
    validation_data = validation_generator,
    validation_steps = len(validation_generator.classes) // BATCH_SIZE,
    callbacks = [lr_reducer, model_checkpoint, early_stopping])


# print validation accuracy from the best model during training 
model = load_model(bst_model_path)

bst_val_acc = max(history.history['val_acc'])
print("Best val acc: {:.1%}".format(bst_val_acc))



