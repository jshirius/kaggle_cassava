# ====================================================
# Dataset
# ====================================================
import torch
from torch.utils.data import Dataset,DataLoader
import cv2




class TrainDataset(Dataset):
    def __init__(self, df, data_root,transform=None):
        self.df = df
        self.file_names = df['image_id'].values
        self.labels = df['label'].values
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
        label = torch.tensor(self.labels[idx]).long()
        return image, label
    

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
        

def prepare_dataloader(df, trn_idx, val_idx, param:dict, get_train_transforms, get_valid_transforms, data_root='../input/cassava-leaf-disease-classification/train_images/'):
    
    #from catalyst.data.sampler import BalanceClassSampler
    
    train_ = df.loc[trn_idx,:].reset_index(drop=True)
    valid_ = df.loc[val_idx,:].reset_index(drop=True)
        
    #train_ds = CassavaDataset(train_, v, transforms=get_train_transforms(), output_label=True, one_hot_label=False, do_fmix=False, do_cutmix=False)
    #valid_ds = CassavaDataset(valid_, data_root, transforms=get_valid_transforms(), output_label=True)
    
    train_ds = TrainDataset(train_, data_root, transform = get_train_transforms())
    valid_ds = TrainDataset(valid_, data_root, transform = get_valid_transforms())


    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=param['train_bs'],
        pin_memory=False,
        drop_last=False,
        shuffle=True,        
        num_workers=param['num_workers'],
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
