import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from sklearn.utils import shuffle
from tqdm import tqdm_notebook
import tensorflow as tf

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
# trainlen = int(len(datamat[:]) * 0.9)
trainlen = 1000
testlen = 100
# print(testlen)
print(trainlen)


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

fig, ax = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle('Histopathologic scans of lymph node sections', fontsize=20)
# Negatives
for i, idx in enumerate(shuffled_data[shuffled_data['label'] == 0]['id'][:5]):
    path = os.path.join(train_path, idx)
    ax[0, i].imshow(readImage(path + '.tif'))
    # Create a Rectangle patch
    box = patches.Rectangle((32, 32), 32, 32, linewidth=4, edgecolor='b', facecolor='none', linestyle=':',
                            capstyle='round')
    ax[0, i].add_patch(box)
ax[0, 0].set_ylabel('Negative samples', size='large')
# Positives
for i, idx in enumerate(shuffled_data[shuffled_data['label'] == 1]['id'][:5]):
    path = os.path.join(train_path, idx)
    ax[1, i].imshow(readImage(path + '.tif'))
    # Create a Rectangle patch
    box = patches.Rectangle((32, 32), 32, 32, linewidth=4, edgecolor='r', facecolor='none', linestyle=':',
                            capstyle='round')
    ax[1, i].add_patch(box)
ax[1, 0].set_ylabel('Tumor tissue samples', size='large')

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


def readCroppedImage2(path, augmentations=False):
    # augmentations parameter is included for counting statistics from images, where we don't want augmentations

    # OpenCV reads the image in bgr format by default
    bgr_img = cv2.imread(path)
    # We flip it to rgb for visualization purposes
    b, g, r = cv2.split(bgr_img)
    rgb_img = cv2.merge([r, g, b])

    if (not augmentations):
        return rgb_img / 255

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
train_images = np.full((trainlen, 32, 32, 3), 0)
train_label = np.zeros(trainlen)
test_images = np.full((testlen, 32, 32, 3), 0)
test_label = np.zeros(testlen)

for j in range(trainlen):
    idx = train_datamat[j][0]
    path = os.path.join(train_path, idx)
    train_images[j] = readCroppedImage(path + '.tif')
    train_label[j] = train_datamat[j][1]
"""for j in range(testlen):
    idxtest = test_datamat[j][0]
    testpath = os.path.join(test_path, idxtest)
    test_images[j] = readCroppedImage(testpath + '.tif')
    test_label[j] = test_datamat[j][1]"""
"""for j in range(trainlen, trainlen + testlen):
    idxtest = train_datamat[j][0]
    testpath = os.path.join(train_path, idxtest)
    test_images[j-trainlen] = readCroppedImage(testpath + '.tif')
    test_label[j-trainlen] = train_datamat[j][1]"""
print('test')
print(test_label.shape)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape=(32, 32, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(16, activation ='relu'))
model.add(tf.keras.layers.Dense(2, activation ='sigmoid'))
model.summary()
model.compile(optimizer = 'adam',
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
"""model.fit(train_images, train_label, epochs=7,
          validation_data =(test_images, test_label))"""
history = model.fit(train_images, train_label, epochs=5, batch_size=10, validation_split=0.1)
print(history.history)
plt.figure()
plt.plot(history.history['loss'], label="testloss")
plt.plot(history.history['val_loss'], label='valloss')
plt.legend()
plt.show()