import os, cv2, random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential
from keras.layers import Dropout, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, Dense, Activation
from keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras import backend as K

import logging

logging.basicConfig(filename='E:/workplace/kaggle/the-nature-conservancy-fisheries-monitoring/log/train1.log', level=logging.INFO)

TRAIN_DIR = 'E:/workplace/kaggle/the-nature-conservancy-fisheries-monitoring/train/train/'
TEST_DIR = 'E:/workplace/kaggle/the-nature-conservancy-fisheries-monitoring/test_stg1/'
FISH_CLASSES = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
ROWS = 90   #720
COLS = 160  #1280
CHANNELS = 3

# 加载图片
def get_images(fish):
    fish_dir = TRAIN_DIR + '{}'.format(fish)
    images = [fish + "/" + im for im in os.listdir(fish_dir)]
    return images

# 读取图片并进行resize
def read_image(src):
    im = cv2.imread(src, cv2.IMREAD_COLOR) # 读取图片，默认为BGR顺序
    im = cv2.resize(im, (COLS, ROWS), interpolation=cv2.INTER_CUBIC) # 缩放图片
    return im

files = []
y_all = []  # 保存所有图片的种类标签

for fish in FISH_CLASSES:
    fish_files = get_images(fish)
    files.extend(fish_files)

    y_fish = np.tile(fish, len(fish_files)) #保存每种类别每张图片的标签
    y_all.extend(y_fish)
    logging.info("{0} photos of {1}".format(len(fish_files), fish))
    print("{0} photos of {1}".format(len(fish_files), fish))  #每个种类的图片数量

# 将list转为array
y_all = np.array(y_all)


X_all = np.ndarray((len(files), ROWS, COLS, CHANNELS), dtype=np.uint8)
# print(X_all)

for i, im in enumerate(files):
    X_all[i] = read_image(TRAIN_DIR + im)
    if i % 1000 == 0:
        print("Processed {} of {}".format(i, len(files)))

logging.info(X_all.shape)
print(X_all.shape)  # (3777, 90, 160, 3)

# One Hot Encoding Labels
y_all = LabelEncoder().fit_transform(y_all)
y_all = np_utils.to_categorical(y_all)

X_train, X_valid, y_train, y_valid = train_test_split(X_all, y_all, test_size=0.2, random_state=23, stratify=y_all)

optimizer = RMSprop(lr=1e-4)
objective = 'categorical_crossentropy'

def center_normalize(x):
    return (x - K.mean(x)) / K.std(x)

model = Sequential()

model.add(Activation(activation=center_normalize, input_shape=(ROWS, COLS, CHANNELS)))

model.add(Convolution2D(32, 5, 5, border_mode='same', activation='relu', dim_ordering='tf'))
model.add(Convolution2D(32, 5, 5, border_mode='same', activation='relu', dim_ordering='tf'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))

model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))

model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))

model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))


model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(len(FISH_CLASSES)))
model.add(Activation('sigmoid'))

model.compile(loss=objective, optimizer=optimizer)

# 当监测值不再改善时，该回调函数将终止训练 (monitor:需要监视的量；patience：当early stop被激活后（如发现loss相比上一个
# epoch训练没有下降）则经过patience个epoch后停止训练；verbose:信息展示模式；mode:'auto','min','max'之一，
# 在min模式下，如果检测值停止下降则中止训练。在max模式下，当检测值不再上升则停止训练。)
early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto')


logging.info("model begin train")
model.fit(X_train, y_train, batch_size=64, nb_epoch=1,
              validation_split=0.2, verbose=1, shuffle=True, callbacks=[early_stopping])

preds = model.predict(X_valid, verbose=1)

logging.info("Validation Log Loss:{}".format(log_loss(y_valid, preds)))
print("Validation Log Loss:{}".format(log_loss(y_valid, preds)))

test_files = [im for im in os.listdir(TEST_DIR)]
test = np.ndarray((len(test_files), ROWS, COLS, CHANNELS), dtype=np.uint8)

for i, im in enumerate(test_files):
    test[i] = read_image(TEST_DIR + im)

test_preds = model.predict(test, verbose=1)

submission = pd.DataFrame(test_preds, columns=FISH_CLASSES)
submission.insert(0, 'image', test_files)
submission.head()
submission.to_csv('E:/workplace/kaggle/the-nature-conservancy-fisheries-monitoring/submissionOne.csv', index=False)
logging.info("end!")
