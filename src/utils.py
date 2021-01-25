from contextlib import contextmanager
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import cv2

@contextmanager
def timer(message: str):
    print(f'[{message}] start.')
    t0 = time.time()
    yield
    elapsed_time = time.time() - t0
    print(f'[{message}] done in {elapsed_time / 60:.1f} min.')


def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def init_logger(log_file='train.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def transform_image_plot(img_path, transform, figsize =(8, 5) ):
    #画像出力用の関数
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Augment an image
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    plt.figure(figsize=figsize)
    plt.imshow(transformed_image);

