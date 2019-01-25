### [residual attention network](https://arxiv.org/pdf/1704.06904.pdf)

#### concept of residual attention
Each Attention Module is divided into two branches: mask branch and trunk branch. The trunk branch performs feature processing and can be adapted to any state-of-the-art network structures.

Given trunk branch output T(x) with input x, the mask branch uses bottom-up top-down structure to learn same size mask M(x) that softly weight output features T(x).

The output mask is used as control gates for neurons of trunk branch similar to Highway Network.

Hi,c(x) = Mi,c(x) ∗ Ti,c(x) 

where i ranges over all spatial positions and c ∈ {1, ..., C} is the index of the channel. 

In Attention Module, each trunk branch has its own mask branch to learn attention that is specialized for its features. 

Similar to ideas in residual learning, if soft mask unit can be constructed as identical mapping, the performances should be no worse than its counterpart without attention. Thus we modify output H of Attention Module as

Hi,c(x) = (1 + Mi,c(x)) ∗ Fi,c(x) 

M(x) ranges from [0, 1], with M(x) approximating 0, H(x) will approximate original features F(x).  <br><br> 


#### imagenet residual attention network 56 structure 
![model structure](https://github.com/yueying-teng/indoor_gps/blob/master/code/Screen%20Shot%202019-01-25%20at%2016.32.20.png)

From input, max pooling are performed several times to increase the receptive field rapidly after a small number of
Residual Units. After reaching the lowest resolution, the global information is then expanded by a symmetrical topdown architecture to guide input features in each position. Linear interpolation up sample the output after some Residual Units. The number of bilinear interpolation is the same as max pooling to keep the output size the same as the input feature map.

Then a sigmoid layer normalizes the output range to [0, 1] after two consecutive 1 × 1 convolution layers. We also added skip connections between bottom-up and top-down parts to capture information from different scales.

We make the size of the smallest output map in each mask branch 7×7 to be consistent with the smallest trunk output map size. Thus 3,2,1 max-pooling layers are used in mask branch with input size 56×56, 28×28, 14×14 respectively. The Attention Module is built by pre-activation Residual Unit.    <br><br>


#### model used in this project
```python
def AttentionResNetCifar10(shape=(32, 32, 3), n_channels=32, n_classes=76):
    '''
    resize input to 32 x 32
    one skip connection is used in the first attention module
    no skip connection is used in the second and third attention modules
    2, 1, 1 max-pooling layers are used in mask branch of each attention modules
    '''
    
    input_ = Input(shape=shape)
    x = Conv2D(n_channels, (5, 5), padding='same')(input_)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)                                        # 16x16

    x = residual_block(x, input_channels=32, output_channels=128)             # 16x16
    x = attention_block_stage2(x)

    x = residual_block(x, input_channels=128, output_channels=256, stride=2)  # 8x8
    x = attention_block_stage3(x)

    x = residual_block(x, input_channels=256, output_channels=512, stride=2)  # 4x4
    x = attention_block_stage3(x)

    x = residual_block(x, input_channels=512, output_channels=1024)
    x = residual_block(x, input_channels=1024, output_channels=1024)
    x = residual_block(x, input_channels=1024, output_channels=1024)

    x = AveragePooling2D(pool_size=(4, 4), strides=(1, 1))(x)                  # 1x1
    x = Flatten()(x)
    output = Dense(n_classes, activation='softmax')(x)

    model = Model(input_, output)
    
    return model
```
