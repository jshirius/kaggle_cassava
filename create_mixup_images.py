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

#mixup後の画像を格納するフォルダ名
folder = "./mixup_alpha_1/"
alpha = 1.0 #ドキュメントによると0.5あたりが良いらしい
batch_size = 32
end_count = 10 #画像はbatch_size * end_count分作成される
num_classes = 5

#csvファイルを読み込む
train = pd.read_csv('../input/cassava-leaf-disease-classification/train.csv' , nrows = 120)

data_root = '../input/cassava-leaf-disease-classification/train_images/'
train_X =[]
train_y = []
for index, row in train.iterrows():
    file_name = row['image_id']
    file_path = f'{data_root}/{file_name}'
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    train_X.append(image)
    train_y.append(row['label'])
    
#numpyに変換する
train_X = np.array(train_X)
train_y = np.array(train_y)
train_y = keras.utils.to_categorical(train_y, num_classes)

#フォルダ作成
os.makedirs(folder, exist_ok=True)

#MixUPを実行する
generator = MixupGenerator(train_X, train_y, alpha=alpha, batch_size=batch_size )()

#画像作成したら随時格納
images_names = []
label_names = []
j_count = 0
for batch_index, (x, y) in enumerate(generator):
    if(end_count < batch_index):
        break

    y = np.argmax(y,  axis = 1)
    for index in range(0, len(y)):

        #画像作成する
        j_name = "mix_1_" + str(j_count) + ".jpg"
        path = folder + j_name

        cv2.imwrite(path, x[index])

        #画像ファイルとラベルを作る
        images_names.append(j_name)
        label_names.append(y[index])

        j_count +=1

#フォルダを圧縮
shutil.make_archive(folder, 'zip', root_dir=folder)

#dfファイル作成
df = pd.DataFrame()
df["image_id"] = images_names
df["label"] = label_names
df.to_csv("mix_train.csv", index = False)

#元のフォルダ削除
shutil.rmtree(folder)

#ラベルの割合表示
vc = df['label'].nunique()
print(vc)
