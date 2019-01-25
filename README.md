# indoor_gps

### data
original image data is extracted from all frames in videos.

these original images are processed so that no identical pairs exist.

80% of the data is used as training data.

### model
transfer learning with Xception as base model. no data augmentation is used. 

testing accuracy: 86%

trained reisdual attention network 56 on resized and zero centered images (32 x 32) from scratch. no data augmentation is used. see model [detail](https://github.com/yueying-teng/indoor_gps/tree/master/code).

testing accuraccy: 100%


