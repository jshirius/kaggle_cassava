# 訓練モデル
#import efficientnet.tfkeras as efn
import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
from tensorflow.keras import optimizers, Sequential, losses, metrics, Model
from tensorflow.keras.callbacks import EarlyStopping
import torch
from torch import nn
import timm

# ====================================================
# MODEL　ResNext
# ====================================================
#https://www.kaggle.com/takiyu/cassava-resnext50-32x4d-starter-training
class CustomResNext(nn.Module):
    def __init__(self, model_name='resnext50_32x4d', pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, CFG.target_size)

    def forward(self, x):
        x = self.model(x)
        return x


#有力
#model_archでモデル(ReXNet,EfficientNetなど)を指定できる
#https://pypi.org/project/timm/
#https://www.kaggle.com/takiyu/pytorch-efficientnet-baseline-train-amp-aug/edit
class CassvaImgClassifier(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        #TODO:dropoutあたり入れてみるか
        self.model.classifier = nn.Linear(n_features, n_class)
        '''
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            #nn.Linear(n_features, hidden_size,bias=True), nn.ELU(),
            nn.Linear(n_features, n_class, bias=True)
        )
        '''
    def forward(self, x):
        x = self.model(x)


        return x
        
        

#https://www.kaggle.com/takiyu/cassava-leaf-disease-training-with-tpu-v2-pods
#EfficientNetB4 tensorflow
def model_fn(input_shape, N_CLASSES):
    inputs = L.Input(shape=input_shape, name='input_image')
    base_model = efn.EfficientNetB4(input_tensor=inputs, 
                                    include_top=False, 
                                    weights='noisy-student', 
                                    pooling='avg')
    base_model.trainable = False

    x = L.Dropout(.5)(base_model.output)
    output = L.Dense(N_CLASSES, activation='softmax', name='output')(x)
    model = Model(inputs=inputs, outputs=output)

    return model