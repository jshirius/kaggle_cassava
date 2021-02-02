# 訓練と評価

import time
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
from src.data_set import TestDataset, LABEL_NUM
from src.model.train_model import CassvaImgClassifier
import os


#https://www.kaggle.com/takiyu/pytorch-efficientnet-baseline-train-amp-aug
#訓練
def train_one_epoch(epoch, config, model, loss_fn, optimizer, train_loader, device, scheduler=None, schd_batch_update=False):
    model.train()

    t = time.time()
    running_loss = None

    scaler = GradScaler()   

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (imgs, image_labels, file_names) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()

        #print(image_labels.shape, exam_label.shape)
        with autocast():
            image_preds = model(imgs)   #output = model(input)
            #print(image_preds.shape)

            loss = loss_fn(image_preds, image_labels)
            
            scaler.scale(loss).backward()

            if running_loss is None:
                running_loss = loss.item()
            else:
                running_loss = running_loss * .99 + loss.item() * .01

            if ((step + 1) %  config['accum_iter'] == 0) or ((step + 1) == len(train_loader)):
                # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad() 
                
                if scheduler is not None and schd_batch_update:
                    scheduler.step()

            if ((step + 1) % config['verbose_step'] == 0) or ((step + 1) == len(train_loader)):
                description = f'epoch {epoch} loss: {running_loss:.4f}'
                
                pbar.set_description(description)
                
    if scheduler is not None and not schd_batch_update:
        scheduler.step()

#https://www.kaggle.com/takiyu/pytorch-efficientnet-baseline-train-amp-aug
# 評価    
def valid_one_epoch(epoch, config, model,loss_fn, val_loader, device, scheduler=None, schd_loss_update=False):
    model.eval()

    t = time.time()
    loss_sum = 0
    sample_num = 0
    image_preds_all = []
    image_targets_all = []
    
    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for step, (imgs, image_labels, file_names) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()
        
        image_preds = model(imgs)   #output = model(input)
        #print(image_preds.shape, exam_pred.shape)
        image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]
        image_targets_all += [image_labels.detach().cpu().numpy()]
        
        loss = loss_fn(image_preds, image_labels)
        
        loss_sum += loss.item()*image_labels.shape[0]
        sample_num += image_labels.shape[0]  

        if ((step + 1) % config['verbose_step'] == 0) or ((step + 1) == len(val_loader)):
            description = f'epoch {epoch} loss: {loss_sum/sample_num:.4f}'
            pbar.set_description(description)
    
    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    accuracy = (image_preds_all==image_targets_all).mean()
    print('validation multi-class accuracy = {:.4f}'.format(accuracy))
    
    if scheduler is not None:
        if schd_loss_update:
            scheduler.step(loss_sum/sample_num)
        else:
            scheduler.step()

    return accuracy

#推論
def inference_one_epoch(model, data_loader, device):
    model.eval()

    image_preds_all = []
    
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, (imgs) in pbar:
        imgs = imgs.to(device).float()
        
        image_preds = model(imgs)   #output = model(input)
        image_preds_all += [torch.softmax(image_preds, 1).detach().cpu().numpy()]
        
    
    image_preds_all = np.concatenate(image_preds_all, axis=0)
    return image_preds_all


def inference_single(model_name, model_root_path, param, transform):
    """ fold対応の推論処理

    Args:
        model_name ([type]): モデル名
        model_root_path ([type]): モデルがあるroot path
        param ([type]): 設定
        transform ([type]): [description]

    Returns:
        [type]: 推論の結果
    """
    
    folds = param["fold_num"]
    tst_preds = []
    for fold in range(folds): 
        # we'll train fold 0 first
        if param["fold_limit"] <= fold:
            break 

        print('Inference fold {} started'.format(fold))

 
        
        test = pd.DataFrame()
        test['image_id'] = list(os.listdir('../input/cassava-leaf-disease-classification/test_images/'))
        test_ds = TestDataset(test, '../input/cassava-leaf-disease-classification/test_images/', transform=transform())

        
        tst_loader = torch.utils.data.DataLoader(
            test_ds, 
            batch_size=param['valid_bs'],
            num_workers=param['num_workers'],
            shuffle=False,
            pin_memory=False,
        )

        device = torch.device(param['device'])
        model = CassvaImgClassifier(model_name, LABEL_NUM).to(device)
        
        tst_preds = []
        
        for i, epoch in enumerate(param['used_epochs']):    
            
            load_path = model_root_path + '{}_fold_{}_{}'.format(model_name, fold, epoch)
            model.load_state_dict(torch.load(load_path))
            
            with torch.no_grad():
                for _ in range(param['tta']):
                    #print(model)
                    tst_preds += [param['weights'][i]/sum(param['weights'])/param['tta']*inference_one_epoch(model, tst_loader, device)]

        #tst_preds = np.mean(tst_preds, axis=0) 
        
        del model
        torch.cuda.empty_cache()
        
    tst_preds = np.mean(tst_preds, axis=0) 
    return tst_preds
    
    
        