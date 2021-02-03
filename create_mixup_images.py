# ====================================================
# MixUP画像作成スクリプト
# ====================================================

import cv2

import pandas as pd
import numpy as np
from src.mixup_generator import MixupGenerator
import os
import matplotlib.pyplot as plt
from PIL import Image
import shutil
import keras
from tqdm import tqdm
import random

#mixup後の画像を格納するフォルダ名
folder = "./mixup_alpha_1/"
alpha = 1.0 #ドキュメントによると0.5あたりが良いらしい
batch_size = 32
end_count = 10 #画像はbatch_size * end_count分作成される
num_classes = 5

#csvファイルを読み込む
train = pd.read_csv('../input/cassava-leaf-disease-classification/train.csv' , nrows = 12000)

data_root = '../input/cassava-leaf-disease-classification/train_images/'

#フォルダ作成
os.makedirs(folder, exist_ok=True)

#ラベル補正
noisy_label = pd.read_csv("./src/data/noisy_label.csv")
#clean labelで推測された方に置き換える
train["label"] = noisy_label["guess_label"]
print("train label clean change")



def get_mixup_data(train, label_id):

    train_X =[]
    train_y = []
    j_count = 0
    for index, row in tqdm(train.iterrows()):
        file_name = row['image_id']
        file_path = f'{data_root}/{file_name}'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        train_X.append(image)

        #擬似的にランダムなラベルを設定する
        train_y.append(random.randrange(num_classes))
        
    #numpyに変換する
    train_X = np.array(train_X)
    train_y = np.array(train_y)
    train_y = keras.utils.to_categorical(train_y, num_classes)

    #MixUPを実行する
    generator = MixupGenerator(train_X, train_y, alpha=alpha, batch_size=batch_size )()


    #画像作成したら随時格納
    images_names = []
    label_names = []
    
    for batch_index, (x, y) in enumerate(generator):
        if(end_count < batch_index):
            break

        y = np.argmax(y,  axis = 1)
        for index in range(0, len(y)):

            #画像作成する
            j_name = "mix_1_" +  str(label_id) + "_"+ str(j_count) + ".jpg"
            path = folder + j_name

            cv2.imwrite(path, x[index])

            #画像ファイルとラベルを作る
            images_names.append(j_name)
            label_names.append(y[index])

            j_count +=1

    #dfデータ作成
    df = pd.DataFrame()
    df["image_id"] = images_names
    df["label"] = label_id

    return df

#画像を作成する
df_0 = train[train['label'] == 0][0:2000]
df_0 = get_mixup_data(df_0, 0)

df_1 = train[train['label'] == 1][0:1400]
df_1 = get_mixup_data(df_1, 1)

df_2 = train[train['label'] == 2][0:1400]
df_2 = get_mixup_data(df_2, 2)

df_3 = train[train['label'] == 3][0:100]
df_3 = get_mixup_data(df_3, 3)

df_4 = train[train['label'] == 4][0:1400]
df_4 = get_mixup_data(df_4, 4)

df = pd.concat([df_0, df_1, df_2, df_3, df_4])
df = df.reset_index(drop=True)

#dfファイル作成
#df = pd.DataFrame()
#df["image_id"] = images_names
#df["label"] = label_names
df.to_csv("mix_train.csv", index = False)


#フォルダを圧縮
shutil.make_archive(folder, 'zip', root_dir=folder)



#元のフォルダ削除
shutil.rmtree(folder)

#ラベルの割合表示
vc = df['label'].nunique()
print(vc)
