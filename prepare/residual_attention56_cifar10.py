
from keras.datasets import cifar10
from residual_attention_network import *
import keras
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping



def AttentionResNetCifar10(shape=(32, 32, 3), n_channels=32, n_classes=10):
    """
    Residual Attention Network 56 for Cifar10 Dataset
    https://arxiv.org/abs/1704.06904
    """
    input_ = Input(shape=shape)
    x = Conv2D(n_channels, (5, 5), padding='same')(input_)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)  # 16x16

    x = residual_block(x, input_channels=32, output_channels=128)  # 16x16
    x = attention_block_stage2(x)

    x = residual_block(x, input_channels=128, output_channels=256, stride=2)  # 8x8
    x = attention_block_stage3(x)

    x = residual_block(x, input_channels=256, output_channels=512, stride=2)  # 4x4
    x = attention_block_stage3(x)

    x = residual_block(x, input_channels=512, output_channels=1024)
    x = residual_block(x, input_channels=1024, output_channels=1024)
    x = residual_block(x, input_channels=1024, output_channels=1024)

    x = AveragePooling2D(pool_size=(4, 4), strides=(1, 1))(x)  # 1x1
    x = Flatten()(x)
    output = Dense(n_classes, activation='softmax')(x)

    model = Model(input_, output)
    
    return model



# load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



# define generators for training and validation data
train_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

val_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
train_datagen.fit(x_train)
val_datagen.fit(x_train)




model = AttentionResNetCifar10(n_classes = 10)

# prepare usefull callbacks
lr_reducer = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=7, min_lr=10e-7, epsilon=0.01, verbose=1)
early_stop= EarlyStopping(monitor='val_acc', min_delta=0, patience=15, verbose=1)
callbacks= [lr_reducer, early_stop]

# define loss, metrics, optimizer
model.compile(keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# fit the model on batches with real-time data augmentation
batch_size = 32

model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=len(x_train)//batch_size, epochs=200,
                    validation_data=val_datagen.flow(x_test, y_test, batch_size=batch_size), 
                    validation_steps=len(x_test)//batch_size,
                    callbacks=callbacks, initial_epoch=0)
                    

