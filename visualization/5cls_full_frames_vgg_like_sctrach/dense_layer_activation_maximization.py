import numpy as np
import scipy.misc
import os
import time
import tempfile
import scipy.misc 
from scipy.ndimage.filters import gaussian_filter, median_filter

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras import backend as K
from keras import activations


# model variables
NUM_CLASSES = 5
IMG_SIZE  = 224

# visualization regularization 
l2decay = 0.01 # typically 0 or 0.0001 or 0.01 or 0.3
blurStd = 0. # typically 0 or 0.5 or 1



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

model.load_weights('./2018-10-18_gps_model.h5')
model.summary()


# https://github.com/raghakot/keras-vis/blob/master/vis/utils/utils.py
def find_layer_idx(model, layer_name):
    """Looks up the layer index corresponding to `layer_name` from `model`.
    Args:
        model: The `keras.models.Model` instance.
        layer_name: The name of the layer to lookup.
    Returns:
        The layer index if found. Raises an exception otherwise.
    """
    layer_idx = None
    for idx, layer in enumerate(model.layers):
        if layer.name == layer_name:
            layer_idx = idx
            break

    if layer_idx is None:
        raise ValueError("No layer with name '{}' within the model".format(layer_name))
    return layer_idx


def apply_modifications(model, custom_objects=None):
    """Applies modifications to the model layers to create a new Graph. For example, simply changing
    `model.layers[idx].activation = new activation` does not change the graph. The entire graph needs to be updated
    with modified inbound and outbound tensors because of change in layer building function.
    Args:
        model: The `keras.models.Model` instance.
    Returns:
        The modified model with changes applied. Does not mutate the original `model`.
    """
    # The strategy is to save the modified model and load it back. This is done because setting the activation
    # in a Keras layer doesnt actually change the graph. We have to iterate the entire graph and change the
    # layer inbound and outbound nodes with modified tensors. This is doubly complicated in Keras 2.x since
    # multiple inbound and outbound nodes are allowed with the Graph API.
    model_path = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()) + '.h5')
    try:
        model.save(model_path)
        return load_model(model_path, custom_objects=custom_objects)
    finally:
        os.remove(model_path)



layer_name = 'dense_2'
# Utility to search for layer index by name. or use model.layers[-1] since this correspond to the last layer of the model
layer_idx = find_layer_idx(model, 'dense_2')

# Swap softmax with linear activation - https://arxiv.org/pdf/1312.6034.pdf
model.layers[layer_idx].activation = activations.linear
# model = utils.apply_modifications(model)
model = apply_modifications(model)

# Specify input and output of the network
input_img = model.layers[0].input
layer_output = model.layers[-1].output




# List of the generated images after learning
kept_images = []
# Update coefficient
learning_rate = 1000


class_index = 2
print('Processing filter %d' % class_index)
start_time = time.time()

# The loss is the activation of the neuron for the chosen class
loss = K.mean(layer_output[:, class_index])


# we compute the gradient of the input picture wrt this loss
grads = K.gradients(loss, input_img)[0]

# this function returns the loss and grads given the input picture, also add a flag to disable the learning phase (in our case dropout)
iterate = K.function([input_img, K.learning_phase()], [loss, grads])

np.random.seed(1337)  # for reproducibility
# start from a gray image with some random noise
input_img_data = np.random.random((1, IMG_SIZE, IMG_SIZE, 3))
input_img_data = (input_img_data - 0.5) * 20 + 128



# run gradient ascent for 1000 steps
for i in range(200):
    loss_value, grads_value = iterate([input_img_data]) 
    input_img_data += grads_value * learning_rate # Apply gradient to image
    
    if i != 1000:
        if l2decay > 0:
            input_img_data *= (1 - l2decay)

    # Gaussian blur
    if blurStd is not 0 and i % blurEvery == 0 :
        input_img_data = gaussian_filter(input_img_data, sigma = [0, 0, blurStd, blurStd]) # blur along H and W but not channels

    if i % 100 == 0:
        print('current loss:', loss_value)
    
    img = deprocess(input_img_data[0])
    kept_images.append((img, loss_value))

end_time = time.time()
print('Filter %d processed in %ds' % (class_index, end_time - start_time))

imsave('%s_class_%d.png' % (layer_name, class_index), img)


