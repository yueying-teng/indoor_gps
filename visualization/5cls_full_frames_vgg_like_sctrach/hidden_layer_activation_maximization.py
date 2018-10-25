
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import skimage.filters
import time
from scipy.misc import imsave
%matplotlib inline

from keras.preprocessing import image
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras import backend as K


NUM_CLASSES = 5
IMG_SIZE  = 224


# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip image tensor to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    
    return x

def normalize(x):
    '''
    util function to normalize a tensor by its L2 norm
    '''
    return x/ (K.sqrt(K.mean(K.square(x))) + K.epsilon())



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

model.load_weights('./models/2018-10-18_gps_model.h5')
model.summary()



layer_name = 'conv2d_1'
leanring_rate = 10


kept_filters = []

for filter_index in range(31):
    # only scan through the first N filters
    print('Processing filter %d' % filter_index)
    
    start_time = time.time()
    
    # built loss function that maximizes the activation of the nth filter of the layer considered
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    
    # compute the gradient of the input picture w.r.t. this loss
    grads = K.gradients(loss, model.input)[0]
    
    # normalize the gradient
    grads = normalize(grads)
    
    # initialize function returns the loss and grads given the input picture
    iterate = K.function([model.input], [loss, grads])
    
    # start from a gray image with some random noise
    np.random.seed(1337)
    input_img_data = np.random.random((1, IMG_SIZE, IMG_SIZE, 3))
    input_img_data = (input_img_data - 0.5) * 20 + 128

    # run gradeint ascent for 50 steps
    for i in range(1000):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * learning_rate
        
        print('current loss:', loss_value)
        if loss_value <= 0.:
            # some filters get stuck at 0, skip them
            break
            
    if loss_value > 0:
        img = deprocess_image(input_img_data[0])
        kept_filters.append((img, loss_value))
        
    end_time = time.time()
    print('Filter %d processed in %ds' % (filter_index, end_time - start_time))
    

# stich the best 36 filters on a 8x8 grid
n = 6
# filters with the highest loss are assumed to be better-looking
kept_filters.sort(key = lambda x: x[1], reverse = True)
kept_filters = kept_filters[: n * n]

# build a black picture with enough space for the 8 x 8 filters of size 128 x 128, with a 5px margin in between
margin = 5
width = n * IMG_SIZE + (n - 1) * margin
height = n * IMG_SIZE + (n - 1) * margin
stitched_filters = np.zeros((width, height, 3))

# fill the picture with our saved filters
for i in range(n):
    for j in range(n):
        img, loss = kept_filters[i * n + j]
        width_margin = (IMG_SIZE + margin) * i
        height_margin = (IMG_SIZE + margin) * j
        stitched_filters[
            width_margin: width_margin + IMG_SIZE,
            height_margin: height_margin + IMG_SIZE, :] = img

# save the result to disk
imsave(('{}_{}x{}_learning_rate{}.png').format(layer_name, n, n, learning_rate), stitched_filters)
