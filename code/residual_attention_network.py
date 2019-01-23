'''
implementation of the mask and trunk branch in attention module in residual attention network
attention_resnet56 is the model for imagenet data implemented in the paper https://arxiv.org/pdf/1704.06904.pdf
three attention modules with 2, 1, 0 skip connections respectively are used 
'''

from keras.layers import BatchNormalization, Conv2D, UpSampling2D, Activation, MaxPool2D
from keras.layers import Add, Multiply, Lambda


def residual_block(input, input_channels = None, output_channels = None, kernel_size = (3, 3), stride = 1):
    '''
    full pre-activation residual block 
    https://arxiv.org/pdf/1603.05027.pdf

    input_channels: number of channels of the input
    output_channels: number of channels of the ouput of this block
    '''
    if output_channels is None:
        output_channels = input.get_shape()[-1].value
    if input_channels is None:
        input_channels = output_channels // 4 # by design
        
    stride = (stride, stride)
    
    x = BatchNormalization()(input)
    x = Activation('relu')(x)
    x = Conv2D(input_channels, (1, 1))(x)
    
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(input_channels, kernel_size, padding = 'same', strides = stride)(x)
    
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(output_channels, (1, 1), padding = 'same')(x)
    
    # when depth of the input and the depth of the processed input do not match
    if input_channels != output_channels or stride != 1:
        input = Conv2D(output_channels, (1, 1), padding = 'same', strides = stride)(input)
        
    x = Add()([x, input])
    
    return x



def attention_block_stage1(input, input_channels = None, output_channels = None):
    '''
    stage1, stage2, stage3 attention modules in 
    Attention-56
    3, 2, 1 max-pooling layers are used in mask branch with input size 56×56, 28×28, 14×14 respectively
    https://arxiv.org/pdf/1704.06904.pdf
    '''
    p = 1 # number of preprocessing residual unit before splitting into mask and branch trunk
    t = 2 # number of residual unit in trunk branch
#     r = 1 # number of residual unit between adjacent pooling layers in mask branch
    
    if input_channels is None:
        input_channels = input.get_shape()[-1].value
    if output_channels is None:
        output_channels = input_channels
        
    for i in range(p):  
        input = residual_block(input)  # preprocessing residual units   # 56*56
    
    # trunk branch
    output_trunk = input   # 56*56 256
    for i in range(t):
        output_trunk = residual_block(output_trunk)
    
    # mask branch
    # downsampling
    output_mask = MaxPool2D(padding = 'same')(input)   # 28*28
    output_mask = residual_block(output_mask)
    
    skip_connection1 = residual_block(output_mask)
    
    output_mask = MaxPool2D(padding = 'same')(output_mask)  # 14*14
    output_mask = residual_block(output_mask)
    
    skip_connection2 = residual_block(output_mask)
    
    output_mask = MaxPool2D(padding = 'same')(output_mask)  # 7*7
    output_mask = residual_block(output_mask)
    
    # upsampling
    output_mask = residual_block(output_mask)
#     output_mask = UpSampling2D(interpolation = 'bilinear')(output_mask)
    output_mask = UpSampling2D()(output_mask)            # 14*14
    
    output_mask = Add()([output_mask, skip_connection2])
    
    output_mask = residual_block(output_mask)
#     output_mask = UpSampling2D(interpolation = 'bilinear')(output_mask)
    output_mask = UpSampling2D()(output_mask)           # 28*28
    
    output_mask = Add()([output_mask, skip_connection1])
    
    output_mask = residual_block(output_mask)   
#     output_mask = UpSampling2D(interpolation = 'bilinear')(output_mask)
    output_mask = UpSampling2D()(output_mask)           # 56*56
    
    output_mask = Conv2D(input_channels, (1, 1))(output_mask)
    output_mask = Conv2D(input_channels, (1, 1))(output_mask)
    output_mask = Activation('sigmoid')(output_mask)
    
    # attention: (1 + output_mask)* output_trunk
    output = Lambda(lambda x: x + 1)(output_mask)
    output = Multiply()([output, output_trunk])
    
    # last residual unit in stage1 attention module
    for i in range(p):
        output = residual_block(output)

    return output



def attention_block_stage2(input, input_channels = None, output_channels = None):
    '''
    stage1, stage2, stage3 attention modules in 
    Attention-56
    3, 2, 1 max-pooling layers are used in mask branch with input size 56×56, 28×28, 14×14 respectively
    https://arxiv.org/pdf/1704.06904.pdf
    '''
    p = 1 # number of preprocessing residual unit before splitting into mask and branch trunk
    t = 2 # number of residual unit in trunk branch
#     r = 1 # number of residual unit between adjacent pooling layers in mask branch
    
    if input_channels is None:
        input_channels = input.get_shape()[-1].value
    if output_channels is None:
        output_channels = input_channels
        
    for i in range(p):  
        input = residual_block(input)  # preprocessing residual units   # 28*28
   
    # trunk branch
    output_trunk = input   # 28*28 512
    for i in range(t):
        output_trunk = residual_block(output_trunk)    
    
    # mask branch
    # downsampling
    output_mask = MaxPool2D(padding = 'same')(input)   # 14*14
    output_mask = residual_block(output_mask)
    
    skip_connection1 = residual_block(output_mask)
    
    output_mask = MaxPool2D(padding = 'same')(output_mask)    # 7*7
    output_mask = residual_block(output_mask)
    
    # upsampling
    output_mask = residual_block(output_mask)
#     output_mask = UpSampling2D(interpolation = 'bilinear')(output_mask)
    output_mask = UpSampling2D()(output_mask)
    
    output_mask = Add()([output_mask, skip_connection1])
    
    output_mask = residual_block(output_mask)
#     output_mask = UpSampling2D(interpolation = 'bilinear')(output_mask)
    output_mask = UpSampling2D()(output_mask)
    
    output_mask = Conv2D(input_channels, (1, 1))(output_mask)
    output_mask = Conv2D(input_channels, (1, 1))(output_mask)
    output_mask = Activation('sigmoid')(output_mask)
    
    # attention: (1 + output_mask)* output_trunk
    output = Lambda(lambda x: x + 1)(output_mask)
    output = Multiply()([output, output_trunk])
    
    # last residual unit in stage1 attention module
    for i in range(p):
        output = residual_block(output)

    return output



def attention_block_stage3(input, input_channels = None, output_channels = None):
    '''
    stage1, stage2, stage3 attention modules in 
    Attention-56
    3, 2, 1 max-pooling layers are used in mask branch with input size 56×56, 28×28, 14×14 respectively
    https://arxiv.org/pdf/1704.06904.pdf
    '''
    p = 1 # number of preprocessing residual unit before splitting into mask and branch trunk
    t = 2 # number of residual unit in trunk branch
#     r = 1 # number of residual unit between adjacent pooling layers in mask branch
    
    if input_channels is None:
        input_channels = input.get_shape()[-1].value
    if output_channels is None:
        output_channels = input_channels
        
    for i in range(p):  
        input = residual_block(input)  # preprocessing residual units   # 14*14

    # trunk branch
    output_trunk = input   # 14*14 1024
    for i in range(t):
        output_trunk = residual_block(output_trunk)     
    
    # mask branch
    # downsampling
    output_mask = MaxPool2D(padding = 'same')(input)   # 7*7
    output_mask = residual_block(output_mask)
    
    # upsampling
    output_mask = residual_block(output_mask)
#     output_mask = UpSampling2D(interpolation = 'bilinear')(output_mask)
    output_mask = UpSampling2D()(output_mask)
    
    output_mask = Conv2D(input_channels, (1, 1))(output_mask)
    output_mask = Conv2D(input_channels, (1, 1))(output_mask)
    output_mask = Activation('sigmoid')(output_mask)
    
    # attention: (1 + output_mask)* output_trunk
    output = Lambda(lambda x: x + 1)(output_mask)
    output = Multiply()([output, output_trunk])
    
    # last residual unit in stage1 attention module
    for i in range(p):
        output = residual_block(output)

    return output





from keras.layers import Input, Conv2D, MaxPool2D, Dense, AveragePooling2D
from keras.layers import Flatten, Activation, BatchNormalization, Dropout
from keras.models import Model
from keras.regularizers import l2


def attention_resnet56(shape = (224, 224, 3), n_channels = 64, n_classes = 76, dropout = 0, regularization = 0.01):
    '''
    https://arxiv.org/pdf/1704.06904.pdf
    '''
    regularizer = l2(regularization)
    
    inp = Input(shape = shape)  # 224*224
    x = Conv2D(n_channels, (7, 7), strides = (2, 2), padding = 'same')(inp)  # 112*112
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)   # 56*56 64
    
    x = residual_block(x, output_channels = n_channels * 4)  # as default strides = (1, 1)   # output shape 56*56 256
    x = attention_block_stage1(x)   # 56*56
    
    x = residual_block(x, output_channels = n_channels * 8, stride = 2)  # stride is changed to 2, 28*28 512
    x = attention_block_stage2(x)   # 28*28
    
    x = residual_block(x, output_channels = n_channels * 16, stride = 2)  # stride is changed to 2, 14*14 1024
    x = attention_block_stage3(x)   # 14*14
    
    x = residual_block(x, output_channels = n_channels * 32, stride = 2)  # 7x7 2048
    x = residual_block(x, output_channels = n_channels * 32)     # 7x7 2048
    x = residual_block(x, output_channels = n_channels * 32)    # 7x7 2048
    
    x = AveragePooling2D(pool_size = (7, 7), strides = (1, 1))(x)  # 1x1 2048
    x = Flatten()(x)
    if dropout:
        x = Dropout(dropout)(x)
    output = Dense(n_classes, kernel_regularizer = regularizer, activation = 'softmax')(x)
                         
    model = Model(inp, output)
                         
    return model



# model = attention_resnet56()
# model.summary()

