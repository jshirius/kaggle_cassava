# ====================================================
# メイン処理
# ====================================================
from src.utils import set_seed
from src.data_set import prepare_dataloader
from src.model.train_model import CassvaImgClassifier
from src.learning import train_one_epoch, valid_one_epoch, inference_single, get_criterion


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
import timm
import cv2

import pandas as pd
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,ToGray,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)

from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

"""
import sys
package_path = '../input/pytorch-image-models/pytorch-image-models-master' #'../input/efficientnet-pytorch-07/efficientnet_pytorch-0.7.0'
sys.path.append(package_path)
"""

#import sys
#sys.path.insert(0,"fmix") 

#設定
CFG = {
    'fold_num': 5,
    'fold_limit': 2, #foldで実際にやるもの fold_num以下
    'seed': 42,
    'model_arch': 'resnext50_32x4d', #resnext50_32x4d #tf_efficientnet_b4_ns #tf_efficientnet_b7_nsはメモリに乗らない #tf_efficientnet_b5_nsはメモリに乗るようだ
    'img_size': 512,
    'epochs': 10, #epochsを10にする
    'train_bs': 32, 
    'valid_bs': 32,
    "drop_rate" : 0.2222, #dropout
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
    'debug': True,
    'train_mode' :True,
    'inference_mode' :True, #internetONだと提出できないので注意が必要
    'inference_model_path' : "./", #推論時のモデルパス
    'tta': 4, #Inference用 どこの
    'used_epochs': [4, 5, 6], #Inference用 どこのepocheを使うか 0始まり
    'weights': [1,1,1] ,#Inference用比率
    "noisy_label_csv" :"./src/data/noisy_label.csv", #ノイズラベル修正用のcsvファイルの場所(ノイズ補正しない場合は空白にする)
    "append_data":"", #  "../input/cassava_append_data",
    "criterion":'LabelSmoothing', # ['CrossEntropyLoss', LabelSmoothing', 'FocalLoss' 'FocalCosineLoss', 'SymmetricCrossEntropyLoss', 'BiTemperedLoss', 'TaylorCrossEntropyLoss'] 損失関数のアルゴリズム
    "smoothing": 0.05,#LabelSmoothingの値
    "target_size":5, #ラベルの数

}

def get_train_transforms():
    return Compose([
            RandomResizedCrop(CFG['img_size'], CFG['img_size']),
            Transpose(p=0.5), #転換
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5), #アフィン変換をランダムに適用します。入力を変換、スケーリング、回転します
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5), #色彩などを変更する
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5), # 輝度
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0), #ピクセル値を255 = 2 ** 8-1で除算し、チャネルごとの平均を減算し、チャネルごとのstdで除算します
            CoarseDropout(p=0.5),#粗いドロップアウト
            Cutout(p=0.5),
            ToGray(p=0.01), #これを反映させたほうがスコアが上がる 0.001上がった
            ToTensorV2(p=1.0),

        ], p=1.)

# 参考に0.9を叩き出したもの
# https://www.kaggle.com/takiyu/cassava-leaf-disease-tpu-v2-pods-inference/
#Pixel-level transforms, Crops(画像の中央領域をトリミング)
# ここから過去のコンペのナレッジ
# https://www.kaggle.com/stonewst98/what-a-pity-only-0-0001-away-from-0-77/notebook
# ToGray
def get_valid_transforms():
    return Compose([
            CenterCrop(CFG['img_size'], CFG['img_size'], p=1.),
            Resize(CFG['img_size'], CFG['img_size']),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)

#推論で使うもの(こちらのほうが、get_valid_transformsよりもスコアが0.005も高い)
#https://www.kaggle.com/takiyu/cassava-resnext50-32x4d-inference?scriptVersionId=52803745
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

#以下のパターンも試す   
# ものすごくスコアが悪くなった    
#def get_test_transforms():
#    return A.Compose([
#            A.Resize(height=img_size, width=img_size, p=1.0),
#            ToTensorV2(p=1.0),
#        ], p=1.0)

if __name__ == '__main__':

    #SEED
    set_seed()

    #訓練データを読み込む
    if(CFG["debug"] == True):
        train = pd.read_csv('../input/cassava-leaf-disease-classification/train.csv' , nrows = 30)
    else:
        train = pd.read_csv('../input/cassava-leaf-disease-classification/train.csv')
    print(train)

    #noisy labelを読み込む
    if(len(CFG["noisy_label_csv"]) > 0):
        noisy_label = pd.read_csv(CFG["noisy_label_csv"])
        #clean labelで推測された方に置き換える
        train["label"] = noisy_label["guess_label"]
        print("train label clean change")

    #追加画像を読み込む
    append_data_dict = None
    if(len(CFG["append_data"]) > 0):
        #訓練データ追加
        p = CFG["append_data"] + "/" + "mix_train.csv"
        append_df = pd.read_csv(p)
        print(append_df)
        train = pd.concat([train, append_df])
        train = train.reset_index(drop=True)

        #image_path, exist_name
        append_data_dict = {}
        append_data_dict['image_path'] = CFG["append_data"] + "/" + "mixup_alpha_1"
        append_data_dict['exist_name'] = "mix"


    if(CFG["train_mode"] == True):
        #ラベルを元に分ける
        folds = StratifiedKFold(n_splits=CFG['fold_num'], shuffle=True, random_state=CFG['seed']).split(np.arange(train.shape[0]), train.label.values)
        print(folds)

        for fold, (trn_idx, val_idx) in enumerate(folds):

            # we'll train fold 0 first
            if CFG["fold_limit"] <= fold:
                break 

            print('Training with {} started'.format(fold))

            print(len(trn_idx), len(val_idx))

            #データのローダーを設定する
            train_loader, val_loader = prepare_dataloader(train, trn_idx, val_idx, CFG, get_train_transforms, get_valid_transforms,  data_root='../input/cassava-leaf-disease-classification/train_images/', append_data_dict = append_data_dict)


            #画像を表示する(デバッグ用普段はコメント化)
            """
            train_iter = iter(train_loader)
            images, label, file_name  = train_iter.next()
            image = images[0]
            img = image[:,:,0]
            plt.imshow(img)
            plt.imsave(file_name[0], img)
            """

            #print(train_data)

            device = torch.device(CFG['device'])
            
            ###########################
            #モデルの読み込み
            ###########################
            model = CassvaImgClassifier(CFG['model_arch'], train.label.nunique(), pretrained=True, drop_rate=CFG["drop_rate"]).to(device)

            #Feature Scaling(正規化)を作成する
            scaler = GradScaler()   
            optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
            #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=CFG['epochs']-1)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG['T_0'], T_mult=1, eta_min=CFG['min_lr'], last_epoch=-1)
            #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=25, 
            #                                                max_lr=CFG['lr'], epochs=CFG['epochs'], steps_per_epoch=len(train_loader))
            
            #損失関数の取得
            criterion = get_criterion(CFG)
            print(f'Criterion: {criterion}')   

            loss_tr = criterion.to(device)
            loss_fn = criterion.to(device)
            #loss_tr = nn.CrossEntropyLoss().to(device) #MyCrossEntropyLoss().to(device)
            #loss_fn = nn.CrossEntropyLoss().to(device)
            
            best_accuracy = 0
            for epoch in range(CFG['epochs']):

                t = "train_one_epoch fold:%s epoch:%s" % ( str(fold), str(epoch))
                print(t)
                train_one_epoch(epoch, CFG, model ,loss_tr, optimizer, train_loader, device, scheduler=scheduler, schd_batch_update=False)

                with torch.no_grad():
                    accuracy = valid_one_epoch(epoch, CFG, model,loss_fn, val_loader, device, scheduler=None, schd_loss_update=False)

                    print("accuracy")
                    print(accuracy)
                    if(best_accuracy < accuracy):
                        t = "best_accuracy_update  accuracy:%s fold:%s epoch:%s" % (str(accuracy), str(fold), str(epoch))
                        print(t)
                        best_accuracy = accuracy
                        torch.save(model.state_dict(),'{}_fold_{}'.format(CFG['model_arch'], fold))

                torch.save(model.state_dict(),'{}_fold_{}_{}'.format(CFG['model_arch'], fold, epoch))
                
            

            #torch.save(model.cnn_model.state_dict(),'{}/cnn_model_fold_{}_{}'.format(CFG['model_path'], fold, CFG['tag']))
            del model, optimizer, train_loader, val_loader, scaler, scheduler
            torch.cuda.empty_cache()

    if(CFG["inference_mode"] == True):
        #推論モード
        #res net
        tst_preds = inference_single("resnext50_32x4d", CFG["inference_model_path"], CFG, get_inference_transforms)
    
        #tf_efficientnet_b4_ns
        #tst_preds = inference_single("tf_efficientnet_b4_ns", "../input/cassava-tf-efficientnet-b4-ns-train/")

        test = pd.DataFrame()
        test['image_id'] = list(os.listdir('../input/cassava-leaf-disease-classification/test_images/'))
        test['label'] = np.argmax(tst_preds, axis=1)
        test.to_csv('submission.csv', index=False)
        test.head()
      