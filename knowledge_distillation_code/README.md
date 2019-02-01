
#### [Knowledge Distillation](https://arxiv.org/pdf/1503.02531.pdf)

The key is to use high temperature softmax, which showcases the dark knowledge learned by the teacher model and generalizes it to the distilled student model. Therefore, the distilled student model can perform better than standalone stundet model on unseen data.

![distill loss](https://github.com/yueying-teng/indoor_gps/blob/master/knowledge_distillation_code/Screen%20Shot%202019-02-01%20at%2016.42.49.png)


Loss used in distillation is a linear combination of both teacher loss and student loss. Better result can be achieved by setting student loss weight lower than the weight of teacher loss.


#### Result 
Teacher model: risdual attention network 56 with 100% testing accuracy and 400MB size 

Standalone student mdoel: transfer learning ResNet 50 with 85% testing accuracy and 100MB size

Distilled model has 95% accuracy on the same testing data and 100MB size 


##### references 
[distillation loss](https://nervanasystems.github.io/distiller/knowledge_distillation/index.html)
