# ====================================================
# 推論メイン処理
# ====================================================

"""
import sys
package_path = '../input/pytorch-image-models/pytorch-image-models-master' #'../input/efficientnet-pytorch-07/efficientnet_pytorch-0.7.0'
sys.path.append(package_path)
sys.path.append("../input/cassava-script")
"""

from src.utils import set_seed
from src.data_set import prepare_dataloader, TestDataset
from src.model.train_model import CassvaImgClassifier
from src.learning import train_one_epoch, valid_one_epoch



from sklearn.model_selection import GroupKFold, StratifiedKFold
import torch
from torch import nn
import os
import torch.nn.functional as F
import sklearn
import warnings
import joblib
from sklearn.metrics import roc_auc_score, log_loss
from sklearn import metrics


import pandas as pd
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)

from albumentations.pytorch import ToTensorV2
from tqdm import tqdm


#設定
CFG = {
    'fold_num': 5,
    'seed': 42,
    'model_arch': 'tf_efficientnet_b4_ns',
    'img_size': 512,
    'epochs': 10,
    'train_bs': 16,
    'valid_bs': 32,
    'T_0': 10,
    'lr': 1e-4,
    'min_lr': 1e-6,
    'weight_decay':1e-6,
    #'num_workers': 4,
    'num_workers': 0, #ローカルPCの設定
    'accum_iter': 2, # suppoprt to do batch accumulation for backprop with effectively larger batch size
    'verbose_step': 1,
    #'device': 'cuda:0'
    'device': 'cpu', #ローカルPCのときの設定
    'tta': 4, #Inference用 どこの
    'used_epochs': [4, 5, 6], #Inference用 どこのepocheを使うか
    'weights': [1,1,1] ,#Inference用比率
}

def get_inference_transforms():
    return Compose([
            RandomResizedCrop(CFG['img_size'], CFG['img_size']),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)
        


if __name__ == '__main__':
     # for training only, need nightly build pytorch
    #意図としてトレーニングしたときのvalの確認をしたい
    set_seed(CFG['seed'])
    
    #訓練データを読み込む
    if(CFG["debug"] == True):
        train = pd.read_csv('../input/cassava-leaf-disease-classification/train.csv' , nrows = 50)
    else:
        train = pd.read_csv('../input/cassava-leaf-disease-classification/train.csv')
    print(train)    
    
    folds = StratifiedKFold(n_splits=CFG['fold_num']).split(np.arange(train.shape[0]), train.label.values)
    for fold, (trn_idx, val_idx) in enumerate(folds): 
        # we'll train fold 0 first
        if fold > 0:
            break 

        print('Inference fold {} started'.format(fold))

        #検証用のデータセットを作成する
        valid_ = train.loc[val_idx,:].reset_index(drop=True)
        #valid_ds = CassavaDataset(valid_, '../input/cassava-leaf-disease-classification/train_images/', transforms=get_inference_transforms(), output_label=False)
        
        #__init__(self, df, data_root, transform=None):
        valid_ds = TestDataset(valid_, '../input/cassava-leaf-disease-classification/train_images/', transform=get_inference_transforms())
     

        
        test = pd.DataFrame()
        test['image_id'] = list(os.listdir('../input/cassava-leaf-disease-classification/test_images/'))
        #test_ds = CassavaDataset(test, '../input/cassava-leaf-disease-classification/test_images/', transforms=get_inference_transforms(), output_label=False)
        test_ds = TestDataset(test, '../input/cassava-leaf-disease-classification/test_images/', transform=get_inference_transforms())
        
        val_loader = torch.utils.data.DataLoader(
            valid_ds, 
            batch_size=CFG['valid_bs'],
            num_workers=CFG['num_workers'],
            shuffle=False,
            pin_memory=False,
        )
        
        tst_loader = torch.utils.data.DataLoader(
            test_ds, 
            batch_size=CFG['valid_bs'],
            num_workers=CFG['num_workers'],
            shuffle=False,
            pin_memory=False,
        )

        device = torch.device(CFG['device'])
        model = CassvaImgClassifier(CFG['model_arch'], train.label.nunique()).to(device)
        
        val_preds = []
        tst_preds = []
        
        #for epoch in range(CFG['epochs']-3):
        for i, epoch in enumerate(CFG['used_epochs']):    
            model.load_state_dict(torch.load('../input/cassava-efficientnet-model/{}_fold_{}_{}'.format(CFG['model_arch'], fold, epoch)))
            
            with torch.no_grad():
                for _ in range(CFG['tta']):
                    #print(model)
                    val_preds += [CFG['weights'][i]/sum(CFG['weights'])/CFG['tta']*inference_one_epoch(model, val_loader, device)]
                    tst_preds += [CFG['weights'][i]/sum(CFG['weights'])/CFG['tta']*inference_one_epoch(model, tst_loader, device)]

        val_preds = np.mean(val_preds, axis=0) 
        tst_preds = np.mean(tst_preds, axis=0) 
        
        print('fold {} validation loss = {:.5f}'.format(fold, log_loss(valid_.label.values, val_preds)))
        print('fold {} validation accuracy = {:.5f}'.format(fold, (valid_.label.values==np.argmax(val_preds, axis=1)).mean()))
        
        del model
        torch.cuda.empty_cache()

