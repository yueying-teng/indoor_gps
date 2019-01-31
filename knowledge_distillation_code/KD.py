
'''
build distilled model that outputs hard_pred and soft_pred in one vector of length NUM_CLASSES* 2
train this distilled model on the same set of training data that is used to train the teacher model and the standalone student model (ResNet50 transfer learning)
test the distilled model on all testing data

'''

import numpy as np
import os
from tqdm import tqdm_notebook
import glob
import datetime

import keras 
from keras.layers import Lambda, concatenate, Activation
from keras.preprocessing import image
from keras.optimizers import Adam, SGD, RMSprop
from keras.models import load_model, Model, Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
import keras.backend as K
from keras.losses import categorical_crossentropy 
from keras.applications.resnet50 import ResNet50, preprocess_input


train_data_dir = '/home/ubuntu/.jupyter/indoor_data/train' 
test_data_dir = '/home/ubuntu/.jupyter/indoor_data/test'

BATCH_SIZE = 16
EPOCHS = 50
IMG_SIZE = 224
NUM_CLASSES = 76


# load true labels in training and testing the distilled model
# both new_test_label and new_train_label should have NUM_CLASSES* 2 number of columns 

new_train_label = np.load('new_train_label.npy')
new_test_label = np.load('new_test_label.npy')



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


hard_label_train, x_train = get_hard_label_and_feature(train_data_dir, IMG_SIZE = IMG_SIZE)
# print(len(hard_label_train))
print(x_train.shape)

hard_label_test, x_test = get_hard_label_and_feature(test_data_dir, IMG_SIZE = IMG_SIZE)
# print(len(hard_label_train))
print(x_test.shape)

np.save('x_train_for_distillation.npy', x_train)
np.save('x_test_for_distillation.npy', x_test)



###  define distilled model that outputs soft_label and soft_pred 

# standalone student model trained on (224, 224, 3) data
student_model = load_model('./student_model/2019-01-30_indoor_gps_student_resnet_model_0.0001.h5')


# remove softmax in student model to output both high temperature and temperature = 1 predictions
# hard_temperature = 1

soft_temperature = 5

student_model.layers.pop()

logits = student_model.layers[-1].output
logits_soft = Lambda(lambda x: x/ 5, name = 'high_temp')(logits)
prob_soft = Activation('softmax', name = 'soft_activation')(logits_soft)

prob_hard = Activation('softmax', name = 'hard_activation')(logits)

output = concatenate([prob_hard, prob_soft])

model = Model(student_model.input, output)



### train the distilled model

# define KD loss function

def kd_loss(y_true, y_pred, alpha):
    '''
    labels and prediction results are stacked in the following order hard label/pred & soft label/pred
    
    '''
    hard_label, soft_label = y_true[:, :NUM_CLASSES], y_true[:, NUM_CLASSES:]
    hard_pred, soft_pred = y_pred[:, :NUM_CLASSES], y_pred[:, NUM_CLASSES:]
    
    hard_loss = categorical_crossentropy(hard_label, hard_pred)
    soft_loss = categorical_crossentropy(soft_label, soft_pred)
    
    loss = alpha * hard_loss + soft_loss
    
    return loss


# fit student model for KD problem

alpha = 0.1

lr = 0.0001
# lr = 0.001
opt = Adam(lr)

lr_reducer = ReduceLROnPlateau(monitor = 'val_acc', factor = 0.2, patience = 7, min_lr = 10e-7, epsilon = 0.01, verbose = 1)
early_stopping = EarlyStopping(monitor = 'val_loss', min_delta = 0.01, patience = 8)

STAMP = "{}_indoor_gps_distilled_model_{}".format(datetime.date.today().strftime("%Y-%m-%d"), lr)
bst_model_path = "./student_model/{}.h5".format(STAMP)

model_checkpoint = ModelCheckpoint(bst_model_path,
                                    monitor = 'val_loss',
                                    save_best_only = True,
                                    save_weights_only = False)



# # cross entropy loss with T = 1 probabilities and targets
# def categorical_crossentropy(y_true, y_pred):

#     y_true = y_true[:, :NUM_CLASSES]
#     y_pred = y_pred[:, :NUM_CLASSES]

#     return categorical_crossentropy(y_true, y_pred)


# # cross entropy loss with soft probabilities and targets
# def soft_logloss(y_true, y_pred):

#     logits = y_true[:, NUM_CLASSES:]
#     y_soft = K.softmax(logits / soft_temperature)
#     y_pred_soft = y_pred[:, NUM_CLASSES:]

#     return categorical_crossentropy(y_soft, y_pred_soft)



model.compile(loss = lambda y_true, y_pred: kd_loss(y_true, y_pred, alpha), optimizer = opt, 
    # metrics = ['accuracy', soft_logloss, categorical_crossentropy]
    metrics = ['accuracy'])

history = model.fit(x_train, new_train_label,
         batch_size = BATCH_SIZE,
         epochs = EPOCHS,
         validation_split = 0.01,
         callbacks = [lr_reducer, model_checkpoint, early_stopping])


# print validation accuracy from the best model during training 

bst_val_acc = max(history.history['val_acc'])
print("Best val acc: {:.1%}".format(bst_val_acc))

    

### test accuracy of the distilled model 

# load the best model just trained above
# custom_objects is used for the customized loss function
model = load_model(bst_model_path, custom_objects = {'<lambda>': lambda y_true, y_pred: y_pred})

y_pred = mdoel.predict(x_test)[:, :NUM_CLASSES] # predcition output when temperature = 1 
y_pred_label = np.argmax(y_pred, axis = 1)
true_test_label = np.argmax(hard_label_test, axis = 1)

right = sum(y_pred_label == true_test_label)
wrong = len(true_test_label) - right

print (right/ len(true_test_label), wrong/ len(true_test_label))

print (right, wrong)

