import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input
from keras import backend as K
path = os.getcwd()

print(path)
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

data_generator = ImageDataGenerator(preprocessing_function = preprocess_input)

model = Sequential()
model.add(tf.keras.applications.ResNet50(
    include_top=False,
    input_shape=(96, 96, 3),
    weights='imagenet',
    input_tensor=None,
    pooling='avg',
    classes=2
))
model.add(tf.keras.layers.Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dense(1, activation = "sigmoid"))
model.layers[0].trainable = False
model.compile(optimizer="sgd",
              loss = 'binary_crossentropy',  metrics=['accuracy',f1_m,precision_m, recall_m])
model.summary()
train_generator = data_generator.flow_from_directory(
    r'.\dataset\transferlearningdata\train', target_size=(96, 96), color_mode='rgb',
    batch_size = 240, class_mode = 'binary'
)
validation_generator = data_generator.flow_from_directory(
    r'.\dataset\transferlearningdata\valid', target_size=(96, 96), color_mode='rgb',
    batch_size = 200, class_mode = 'binary')

history = model.fit_generator(train_generator, steps_per_epoch = 30,
                    validation_data = validation_generator,  validation_steps=4, epochs = 15)

plt.figure()
plt.plot(history.history['loss'], label="training loss")
plt.plot(history.history['val_loss'], label='test loss')
plt.legend()
plt.show()
plt.figure()
plt.plot(history.history['acc'], label="training accuracy")
plt.plot(history.history['val_acc'], label='test accuracy')
plt.legend()
plt.show()
plt.figure()
plt.plot(history.history['f1_m'], label="training F1")
plt.plot(history.history['val_f1_m'], label='test F1')
plt.legend()
plt.show()