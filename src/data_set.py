# ====================================================
# Dataset
# ====================================================
import torch
from torch.utils.data import Dataset,DataLoader
import cv2
import numpy as np

#Mixup, Cutmix, FMix Visualisations
#from fmix.fmix import sample_mask, make_low_freq_image, binarise_mask

#ラベルの最大数
LABEL_NUM = 5

class TrainDataset(Dataset):
    def __init__(self, df, data_root, append_data_dict,transform=None):
        """[summary]

        Args:
            df ([type]): [description]
            data_root ([type]): [description]
            append_data_dict ([dict]): image_path, exist_name
            transform ([type], optional): [description]. Defaults to None.
        """
        self.df = df
        self.file_names = df['image_id'].values
        self.labels = df['label'].values
        self.transform = transform
        self.data_root = data_root
        self.append_data_dict = append_data_dict
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = f'{self.data_root}/{file_name}'

        #appendの確認
        if(self.append_data_dict != None):
            #appendデータ
            if(self.append_data_dict['exist_name'] in file_name):
                file_path = self.append_data_dict["image_path"] + "/" + file_name
            
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        label = torch.tensor(self.labels[idx]).long()
        return image, label, file_name
    

class TestDataset(Dataset):
    def __init__(self, df, data_root, transform=None):
        self.df = df
        self.file_names = df['image_id'].values
        self.transform = transform
        self.data_root = data_root
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = f'{self.data_root}/{file_name}'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image
        

def prepare_dataloader(df, trn_idx, val_idx, param:dict, get_train_transforms, get_valid_transforms, data_set_mode = 1, data_root='../input/cassava-leaf-disease-classification/train_images/', append_data_dict:dict=None):
    
    #from catalyst.data.sampler import BalanceClassSampler
    
    train_ = df.loc[trn_idx,:].reset_index(drop=True)
    valid_ = df.loc[val_idx,:].reset_index(drop=True)

    if(data_set_mode == 1):    
        train_ds = TrainDataset(train_, data_root, append_data_dict,transform = get_train_transforms())
        valid_ds = TrainDataset(valid_, data_root, append_data_dict,transform = get_valid_transforms())
    else:
        #from fmixが必要
        train_ds = CassavaDataset(train_, data_root, param, transforms=get_train_transforms(), output_label=True, one_hot_label=False, do_fmix=False, do_cutmix=False)
        valid_ds = CassavaDataset(valid_, data_root, param, transforms=get_valid_transforms(), output_label=True)
    

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=param['train_bs'],
        pin_memory=False,
        drop_last=False,
        shuffle=True,        
        num_workers=param['num_workers'],
        collate_fn=param['collate'], #cutmixのために追加
        #sampler=BalanceClassSampler(labels=train_['label'].values, mode="downsampling")
    )
    val_loader = torch.utils.data.DataLoader(
        valid_ds, 
        batch_size=param['valid_bs'],
        num_workers=param['num_workers'],
        shuffle=False,
        pin_memory=False,
    )
    return train_loader, val_loader


#from
#https://www.kaggle.com/khyeh0719/pytorch-efficientnet-baseline-train-amp-aug
def rand_bbox(size, lam):
    W = size[0]
    H = size[1]
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

def get_img(path):
    im_bgr = cv2.imread(path)
    im_rgb = im_bgr[:, :, ::-1]
    #print(im_rgb)
    return im_rgb

class CassavaDataset(Dataset):
    def __init__(self, df, data_root,param, 
                 transforms=None, 
                 output_label=True, 
                 one_hot_label=False,
                 do_fmix=False, 
                 do_cutmix=False,
                 cutmix_params={
                     'alpha': 1,
                 }
                ):
        
        super().__init__()

        fmix_params={
            'alpha': 1., 
            'decay_power': 3., 
            'shape': (param['img_size'], param['img_size']),
            'max_soft': True, 
            'reformulate': False
        },

        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root
        self.do_fmix = do_fmix
        self.fmix_params = fmix_params
        self.do_cutmix = do_cutmix
        self.cutmix_params = cutmix_params
        
        self.output_label = output_label
        self.one_hot_label = one_hot_label
        
        if output_label == True:
            self.labels = self.df['label'].values
            #print(self.labels)
            
            if one_hot_label is True:
                self.labels = np.eye(self.df['label'].max()+1)[self.labels]
                #print(self.labels)
            
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index: int):
        
        # get labels
        if self.output_label:
            target = self.labels[index]
          
        img  = get_img("{}/{}".format(self.data_root, self.df.loc[index]['image_id']))

        if self.transforms:
            img = self.transforms(image=img)['image']
        
        if self.do_fmix and np.random.uniform(0., 1., size=1)[0] > 0.5:
            with torch.no_grad():
                #lam, mask = sample_mask(**self.fmix_params)
                
                lam = np.clip(np.random.beta(self.fmix_params['alpha'], self.fmix_params['alpha']),0.6,0.7)
                
                # Make mask, get mean / std
                mask = make_low_freq_image(self.fmix_params['decay_power'], self.fmix_params['shape'])
                mask = binarise_mask(mask, lam, self.fmix_params['shape'], self.fmix_params['max_soft'])
    
                fmix_ix = np.random.choice(self.df.index, size=1)[0]
                fmix_img  = get_img("{}/{}".format(self.data_root, self.df.iloc[fmix_ix]['image_id']))

                if self.transforms:
                    fmix_img = self.transforms(image=fmix_img)['image']

                mask_torch = torch.from_numpy(mask)
                
                # mix image
                img = mask_torch*img+(1.-mask_torch)*fmix_img

                #print(mask.shape)

                #assert self.output_label==True and self.one_hot_label==True

                # mix target
                rate = mask.sum()/CFG['img_size']/CFG['img_size']
                target = rate*target + (1.-rate)*self.labels[fmix_ix]
                #print(target, mask, img)
                #assert False
        
        if self.do_cutmix and np.random.uniform(0., 1., size=1)[0] > 0.5:
            #print(img.sum(), img.shape)
            with torch.no_grad():
                cmix_ix = np.random.choice(self.df.index, size=1)[0]
                cmix_img  = get_img("{}/{}".format(self.data_root, self.df.iloc[cmix_ix]['image_id']))
                if self.transforms:
                    cmix_img = self.transforms(image=cmix_img)['image']
                    
                lam = np.clip(np.random.beta(self.cutmix_params['alpha'], self.cutmix_params['alpha']),0.3,0.4)
                bbx1, bby1, bbx2, bby2 = rand_bbox((CFG['img_size'], CFG['img_size']), lam)

                img[:, bbx1:bbx2, bby1:bby2] = cmix_img[:, bbx1:bbx2, bby1:bby2]

                rate = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (CFG['img_size'] * CFG['img_size']))
                target = rate*target + (1.-rate)*self.labels[cmix_ix]
                
            #print('-', img.sum())
            #print(target)
            #assert False
                            
        # do label smoothing
        #print(type(img), type(target))
        if self.output_label == True:
            return img, target
        else:
            return img