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
# Label Smoothing
# ====================================================
# From
# https://www.kaggle.com/piantic/train-cassava-starter-using-various-loss-funcs
class LabelSmoothingLoss(nn.Module): 
    def __init__(self, classes=5, smoothing=0.0, dim=-1): 
        super(LabelSmoothingLoss, self).__init__() 
        self.confidence = 1.0 - smoothing 
        self.smoothing = smoothing 
        self.cls = classes 
        self.dim = dim 
    def forward(self, pred, target): 
        pred = pred.log_softmax(dim=self.dim) 
        with torch.no_grad():
            true_dist = torch.zeros_like(pred) 
            true_dist.fill_(self.smoothing / (self.cls - 1)) 
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence) 
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

# ====================================================
# TaylorCrossEntropyLoss
# ====================================================
# From 
# https://www.kaggle.com/piantic/train-cassava-starter-using-various-loss-funcs
class TaylorSoftmax(nn.Module):
    '''
    This is the autograd version
    '''
    def __init__(self, dim=1, n=2):
        super(TaylorSoftmax, self).__init__()
        assert n % 2 == 0
        self.dim = dim
        self.n = n

    def forward(self, x):
        '''
        usage similar to nn.Softmax:
            >>> mod = TaylorSoftmax(dim=1, n=4)
            >>> inten = torch.randn(1, 32, 64, 64)
            >>> out = mod(inten)
        '''
        fn = torch.ones_like(x)
        denor = 1.
        for i in range(1, self.n+1):
            denor *= i
            fn = fn + x.pow(i) / denor
        out = fn / fn.sum(dim=self.dim, keepdims=True)
        return out

    
class TaylorCrossEntropyLoss(nn.Module):
    def __init__(self, n=2, ignore_index=-1, reduction='mean', smoothing=0.05):
        super(TaylorCrossEntropyLoss, self).__init__()
        assert n % 2 == 0
        self.taylor_softmax = TaylorSoftmax(dim=1, n=n)
        self.reduction = reduction
        self.ignore_index = ignore_index

        #ラベルは５つと決まっているので5にした
        self.lab_smooth = LabelSmoothingLoss(5, smoothing=smoothing)
        #self.lab_smooth = LabelSmoothingLoss(CFG.target_size, smoothing=smoothing)

    def forward(self, logits, labels):
        log_probs = self.taylor_softmax(logits).log()
        #loss = F.nll_loss(log_probs, labels, reduction=self.reduction,
        #        ignore_index=self.ignore_index)
        loss = self.lab_smooth(log_probs, labels)
        return loss

# ====================================================
# MODEL　ResNext
# ====================================================
#https://www.kaggle.com/takiyu/cassava-resnext50-32x4d-starter-training
#現状、CassvaImgClassifierとほぼ同じ処理なので、以下の関数は利用しなくて良い
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
    def __init__(self, model_arch, n_class, pretrained=False, drop_rate = 0.0):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained, drop_rate= drop_rate)

        if("resnext" in model_arch):
            #resnextの場合
            n_features = self.model.fc.in_features
            self.model.fc = nn.Linear(n_features, n_class)
        else:    
            #EfficientNetなど
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