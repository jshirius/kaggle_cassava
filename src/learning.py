# 訓練と評価

import time
from tqdm import tqdm
import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
from src.data_set import TestDataset, LABEL_NUM
from src.model.train_model import CassvaImgClassifier, LabelSmoothingLoss, TaylorCrossEntropyLoss, CutMixCriterion, TaylorSmoothedLoss
import os
from fmix import sample_mask


def get_criterion(config, criterion_name=""):
    if config["criterion"] =='CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    elif config["criterion"] =='LabelSmoothing':
        criterion = LabelSmoothingLoss(classes=config['target_size'], smoothing=config['smoothing'])
    elif config["criterion"] =='FocalLoss':
        criterion = FocalLoss().to(device)
    elif config["criterion"] =='FocalCosineLoss':
        criterion = FocalCosineLoss()
    elif config["criterion"] =='SymmetricCrossEntropyLoss':
        criterion = SymmetricCrossEntropy().to(device)
    elif config["criterion"] =='BiTemperedLoss': 
        criterion = BiTemperedLogisticLoss(t1=CFG.t1, t2=CFG.t2, smoothing=CFG.smoothing)
    elif config["criterion"] =='TaylorCrossEntropyLoss':
        criterion = TaylorCrossEntropyLoss(smoothing=config['smoothing'])
    elif config["criterion"] =='TaylorSmoothedLoss':
        criterion = TaylorSmoothedLoss(smoothing=config['smoothing'])
    elif criterion_name == 'CutMix':
        criterion = CutMixCriterion(get_criterion(config["criterion"]))        
    return criterion


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix_single(data, target, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    lam = np.clip(np.random.beta(alpha, alpha),0.3,0.4)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    new_data = data.clone()
    new_data[:, :, bby1:bby2, bbx1:bbx2] = data[indices, :, bby1:bby2, bbx1:bbx2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    targets = (target, shuffled_target, lam)

    return new_data, targets

def fmix(device, data, targets, alpha, decay_power, shape, max_soft=0.0, reformulate=False):
    lam, mask = sample_mask(alpha, decay_power, shape, max_soft, reformulate)
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]
    x1 = torch.from_numpy(mask).to(device)*data
    x2 = torch.from_numpy(1-mask).to(device)*shuffled_data
    targets=(targets, shuffled_targets, lam)
    
    return (x1+x2), targets


def cutmix(batch):

    #print(batch[0])
    #print(batch[0][2])
 

    img_size = 512 #ハードコーディング

    batch_size = len(batch)
    data = np.zeros((batch_size, 3, img_size, img_size))
    targets = np.zeros((batch_size))
    file_names = [""] * batch_size
    for i in range(batch_size):
        data[i,:,:,:] = batch[i][0]
        targets[i] = batch[i][1]
        file_names[i] = batch[i][2]

    indices = torch.randperm(batch_size)
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]
    

    lam = np.random.beta(1 , 1)

    image_h, image_w = data.shape[2:]
    cx = np.random.uniform(0, image_w)
    cy = np.random.uniform(0, image_h)
    w = image_w * np.sqrt(1 - lam)
    h = image_h * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, image_w)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, image_h)))

    data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]
    return_targets = torch.zeros((batch_size,3),dtype=torch.int64)
    return_targets[:,0] = torch.from_numpy(targets)
    return_targets[:,1] = torch.from_numpy(shuffled_targets)
    return_targets[0,2] = lam

    #print(return_targets)
    #return_filename = torch.zeros((batch_size,3),dtype=torch.int64)
    #return_filename[:,0] = torch.from_numpy(file_names)
    #return_filename[:,1] = torch.from_numpy(shuffled_file_names)
    #return_filename[0,2] = lam

    #file_namesはダミー

    return torch.from_numpy(data), return_targets, file_names
        
class CutMixCollator:
    def __call__(self, batch):
        #batch = torch.utils.data.dataloader.default_collate(batch)
        batch = cutmix(batch)
        return batch


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

        #cutmixの対応
        use_cutmix = False
        if("use_cutmix" in config  and config["use_cutmix"] == True):
            mix_decision = np.random.rand()
            #mix_decision = 0.1
            if(mix_decision < 0.25):
                t = "use_cutmix  step:%d" % step
                #print(t)
                imgs, image_labels = cutmix_single(imgs, image_labels, 1.)
                use_cutmix = True
            elif(mix_decision >=0.25 and mix_decision < 0.5):
                t = "use_fmix  step:%d" % step
                #print(t)
                imgs, image_labels = fmix(device, imgs, image_labels, alpha=1., decay_power=5., shape=(512,512))
                use_cutmix = True

        #print(image_labels.shape, exam_label.shape)
        with autocast():
            image_preds = model(imgs.float())   #output = model(input)
            #print(image_preds.shape)

            #loss = loss_fn(image_preds, image_labels)
            if(use_cutmix == True):
                #cutmix用
                loss = loss_fn(image_preds, image_labels[0]) * image_labels[2] + loss_fn(image_preds, image_labels[1]) * (1. - image_labels[2])
            else:
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
        
        #tst_preds = []
        
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
    
    
        