

import numpy as np
import pandas as pd
import os
import datetime

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from keras.optimizers import Adam, SGD, RMSprop
from keras.models import Model, Input, load_model
from keras.applications import xception
from keras.applications.xception import preprocess_input
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler





IMG_SIZE = 299
img_shape = (IMG_SIZE, IMG_SIZE, 3)

train_data_dir = '/home/ubuntu/.jupyter/indoor_data/train' 
validation_data_dir = '/home/ubuntu/.jupyter/indoor_data/test'

EPOCHS = 50
BATCH_SIZE = 16
NUM_CLASSES = 76



# create the base pre-trained model
base_model = xception.Xception(weights = 'imagenet', include_top = False)

# add global average pooling layer
x = base_model.output
x = BatchNormalization()(x)
x = GlobalAveragePooling2D()(x)

# add fully connected layer
x = Dropout(0.5)(x)
# x = Dropout(0.6)(x)

x = Dense(512, activation = 'relu')(x)
# x = Dense(1024, activation = 'relu')(x)

# x = Dropout(0.7)(x)
x = Dropout(0.5)(x)

# classification layer
predictions = Dense(NUM_CLASSES, activation = 'softmax')(x)

# model
model = Model(inputs = base_model.input, outputs = predictions)
model.summary()





# fine tune - chose to train the top 2 xception blocks, freeze the first 116 layers and unfreeze the rest:
# for layer in model.layers[:116]:
#     layer.trainable = False
# for layer in model.layers[116:]:
#     layer.trainable = True

for layer in base_model.layers:
    layer.trainable = False
    

# lr = 0.0001 - fine tune learning rate 
# learning rate decay with number of epochs 

lr = 0.01
def scheduler(epoch):
    return lr * 0.9 ** epoch

lr_scheduler = LearningRateScheduler(scheduler)


early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10)
STAMP = "{}_fulldata_gps_xception_model_{}".format(datetime.date.today().strftime("%Y-%m-%d"), lr)

bst_model_path = "/home/ubuntu/.jupyter/indoor_models/{}.h5".format(STAMP)
model_checkpoint = ModelCheckpoint(bst_model_path,
                                    monitor = 'val_loss',
                                    save_best_only = True,
                                    save_weights_only = False)


# fine tune - need to recompile the model for the modifications, lower learning rate & different optimizer, to take effect
# model.compile(optimizer = RMSprop(lr = lr, rho = 0.9), loss = 'categorical_crossentropy',
#               metrics = ["accuracy"])

model.compile(optimizer = Adam(lr = lr, decay = 1e-6), loss = 'categorical_crossentropy', metrics = ["accuracy"])




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


history = model.fit_generator(
    train_generator,
    steps_per_epoch = len(train_generator.classes) // BATCH_SIZE,
    epochs = EPOCHS,
    validation_data = validation_generator,
    validation_steps = len(validation_generator.classes) // BATCH_SIZE,
    callbacks = [lr_scheduler, model_checkpoint, early_stopping])


# print validation accuracy from the best model during training 
model = load_model(bst_model_path)

bst_val_acc = max(history.history['val_acc'])
print("Best val acc: {:.1%}".format(bst_val_acc))

