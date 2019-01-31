
'''
transfer learn ResNet50 as the student model
test this standalone student model on all testing data 

'''


import numpy as np
import os
import datetime

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout, Activation
from keras.models import load_model, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.applications.resnet50 import ResNet50, preprocess_input



IMG_SIZE = 224
img_shape = (IMG_SIZE, IMG_SIZE, 3)

train_data_dir = '/home/ubuntu/.jupyter/indoor_data/train' 
validation_data_dir = '/home/ubuntu/.jupyter/indoor_data/test'


EPOCHS = 50
BATCH_SIZE = 16
NUM_CLASSES = 76



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




base_model = ResNet50(include_top = False, weights = 'imagenet')

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
x = Dense(NUM_CLASSES)(x)
predictions = Activation('softmax')(x)

# model
model = Model(inputs = base_model.input, outputs = predictions)
model.summary()


for layer in base_model.layers:
    layer.trainable = False



lr = 0.0001
# lr = 0.001
lr_reducer = ReduceLROnPlateau(monitor = 'val_acc', factor = 0.2, patience = 7, min_lr = 10e-7, epsilon = 0.01, verbose = 1)
early_stopping = EarlyStopping(monitor = 'val_loss', min_delta = 0.01, patience = 8)

STAMP = "{}_indoor_gps_student_resnet_model_{}".format(datetime.date.today().strftime("%Y-%m-%d"), lr)
bst_model_path = "./student_model/{}.h5".format(STAMP)

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


    

