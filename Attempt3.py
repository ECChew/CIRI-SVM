from keras import backend as K
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from tqdm import tqdm_notebook
import tensorflow as tf
from sklearn.utils import shuffle

data = pd.read_csv('dataset/train_labels.csv')
testdata = pd.read_csv('dataset/sample_submission.csv')
train_path = 'dataset/train/'
test_path = 'dataset/test/'
# quick look at the label stats
data['label'].value_counts()
print(type(data['label'][0]))
train_datamat = data.to_numpy()
test_datamat = testdata.to_numpy()
print(train_datamat.shape)
trainlen = int(len(train_datamat[:]))
# trainlen = int(1e5)
testlen = int(1e4)


# %%
def getLabels():
    data = pd.read_csv('dataset/train_labels.csv')


def readImage(path):
    # OpenCV reads the image in bgr format by default
    bgr_img = cv2.imread(path)
    # We flip it to rgb for visualization purposes
    b, g, r = cv2.split(bgr_img)
    rgb_img = cv2.merge([r, g, b])
    return rgb_img


# random sampling
shuffled_data = shuffle(data)

# fig, ax = plt.subplots(2, 5, figsize=(20, 8))
# fig.suptitle('Histopathologic scans of lymph node sections', fontsize=20)
# # Negatives
# for i, idx in enumerate(shuffled_data[shuffled_data['label'] == 0]['id'][:5]):
#     path = os.path.join(train_path, idx)
#     ax[0, i].imshow(readImage(path + '.tif'))
#     # Create a Rectangle patch
#     box = patches.Rectangle((32, 32), 32, 32, linewidth=4, edgecolor='b', facecolor='none', linestyle=':',
#                             capstyle='round')
#     ax[0, i].add_patch(box)
# ax[0, 0].set_ylabel('Negative samples', size='large')
# # Positives
# for i, idx in enumerate(shuffled_data[shuffled_data['label'] == 1]['id'][:5]):
#     path = os.path.join(train_path, idx)
#     ax[1, i].imshow(readImage(path + '.tif'))
#     # Create a Rectangle patch
#     box = patches.Rectangle((32, 32), 32, 32, linewidth=4, edgecolor='r', facecolor='none', linestyle=':',
#                             capstyle='round')
#     ax[1, i].add_patch(box)
# ax[1, 0].set_ylabel('Tumor tissue samples', size='large')

# %%
"""
##### Essai pour explorer les données
print(i,idx)
print(readImage(path + '.tif').shape)
print(readImage(path + '.tif')[0,2,:])
myCmd = 'du '+ path + '.tif'
"""

# %%

import random

ORIGINAL_SIZE = 96  # original size of the images - do not change

# AUGMENTATION VARIABLES
CROP_SIZE = 32  # final size after crop
RANDOM_ROTATION = 0  # 3    # range (0-180), 180 allows all rotation variations, 0=no change
RANDOM_SHIFT = 0  # 2        # center crop shift in x and y axes, 0=no change. This cannot be more than (ORIGINAL_SIZE - CROP_SIZE)//2
RANDOM_BRIGHTNESS = 0  # 7  # range (0-100), 0=no change
RANDOM_CONTRAST = 0  # 5    # range (0-100), 0=no change
RANDOM_90_DEG_TURN = 0  # 1  # 0 or 1= random turn to left or right


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def readCroppedImage(path, augmentations=False):
    # augmentations parameter is included for counting statistics from images, where we don't want augmentations

    # OpenCV reads the image in bgr format by default
    bgr_img = cv2.imread(path)
    # We flip it to rgb for visualization purposes
    b, g, r = cv2.split(bgr_img)
    rgb_img = cv2.merge([r, g, b])

    if (not augmentations):
        # crop to center and normalize to 0-1 range
        start_crop = (ORIGINAL_SIZE - CROP_SIZE) // 2
        end_crop = start_crop + CROP_SIZE
        rgb_img = rgb_img[(start_crop):(end_crop), (start_crop):(end_crop)] / 255
        return rgb_img

        # random rotation
    rotation = random.randint(-RANDOM_ROTATION, RANDOM_ROTATION)
    if (RANDOM_90_DEG_TURN == 1):
        rotation += random.randint(-1, 1) * 90
    M = cv2.getRotationMatrix2D((48, 48), rotation, 1)  # the center point is the rotation anchor
    rgb_img = cv2.warpAffine(rgb_img, M, (96, 96))

    # random x,y-shift
    x = random.randint(-RANDOM_SHIFT, RANDOM_SHIFT)
    y = random.randint(-RANDOM_SHIFT, RANDOM_SHIFT)

    # crop to center and normalize to 0-1 range
    start_crop = (ORIGINAL_SIZE - CROP_SIZE) // 2
    end_crop = start_crop + CROP_SIZE
    rgb_img = rgb_img[(start_crop + x):(end_crop + x), (start_crop + y):(end_crop + y)] / 255

    # Random flip
    flip_hor = bool(random.getrandbits(1))
    flip_ver = bool(random.getrandbits(1))
    if (flip_hor):
        rgb_img = rgb_img[:, ::-1]
    if (flip_ver):
        rgb_img = rgb_img[::-1, :]

    # Random brightness
    br = random.randint(-RANDOM_BRIGHTNESS, RANDOM_BRIGHTNESS) / 100.
    rgb_img = rgb_img + br

    # Random contrast
    cr = 1.0 + random.randint(-RANDOM_CONTRAST, RANDOM_CONTRAST) / 100.
    rgb_img = rgb_img * cr

    # clip values to 0-1 range
    rgb_img = np.clip(rgb_img, 0, 1.0)
    return rgb_img


# %%
"""
fig, ax = plt.subplots(1,5, figsize=(20,4))
fig.suptitle('Random augmentations to the same image',fontsize=20)
# Negatives
for i, idx in enumerate(shuffled_data[shuffled_data['label'] == 0]['id'][:1]):
    for j in range(5):
        path = os.path.join(train_path, idx)
        ax[j].imshow(readCroppedImage(path + '.tif'))
        if j == 4 :
            print(readCroppedImage(path + '.tif'))
"""
# %%

# train_images = np.full((trainlen, 32, 32, 3), 0)
# train_label = np.zeros(trainlen)
# test_images = np.full((testlen, 32, 32, 3), 0)
# test_label = np.zeros(testlen)
#
# for j in range(trainlen):
#     idx = train_datamat[j][0]
#     path = os.path.join(train_path, idx)
#     train_images[j] = readCroppedImage(path + '.tif')
#     train_label[j] = train_datamat[j][1]
# for j in range(testlen):
#     idxtest = test_datamat[j][0]
#     testpath = os.path.join(test_path, idxtest)
#     test_images[j] = readCroppedImage(testpath + '.tif')
#     test_label[j] = test_datamat[j][1]
#
# np.save(r'dataset\train_images_full.npy', train_images)
# np.save(r'dataset\train_label_full.npy', train_label)
# np.save(r'dataset\test_images_full.npy', test_images)
# np.save(r'dataset\test_label_full.npy', test_label)

train_images = np.load(r'dataset\train_images_full.npy')
train_label = np.load(r'dataset\train_label_full.npy')
test_images = np.load(r'dataset\test_images_full.npy')
test_label = np.load(r'dataset\test_label_full.npy')

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(64, (2, 2), padding='same', input_shape=(32, 32, 3)))
model.add(tf.keras.layers.Conv2D(64, (2, 2), padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(rate=0.2))
model.add(tf.keras.layers.Conv2D(128, (2, 2), padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(rate=0.2))
model.add(tf.keras.layers.Conv2D(256, (2, 2), padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(rate=0.2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(2048, activation='relu', use_bias=True))
model.add(tf.keras.layers.Dropout(rate=0.4))
model.add(tf.keras.layers.Dense(1, activation='sigmoid', use_bias=True))
model.summary()

model.compile(optimizer=tf.keras.optimizers.SGD(),
              loss='binary_crossentropy',
              metrics=['accuracy', f1_m])

history = model.fit(train_images, train_label, epochs=50, batch_size=256, validation_split=0.2)
print('len train_images :', len(train_images))
print(history.history)
print(model)
# plt.figure()
# plt.plot(history.history['loss'], label="trainloss")
# plt.plot(history.history['val_loss'], label='valloss')
# plt.legend()
# plt.show()
plt.figure()
plt.title('')
plt.plot(history.history['accuracy'], label="trainacc")
plt.plot(history.history['val_accuracy'], label='valacc')
plt.legend()
plt.show()
