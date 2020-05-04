#Import scikit-learn dataset library
from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
pca = PCA(n_components=2)


# Import train_test_split function
from sklearn.model_selection import train_test_split
train_images = np.load(r'dataset\train_images_1k.npy')
train_label = np.load(r'dataset\train_label_1k.npy')
train_images = train_images.reshape(1000, 32*32*3)


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(train_images,train_label, test_size=0.1,random_state=109) # 90% training and 10% test

#Import svm model
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
acc = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# Model Precision: what percentage of positive tuples are labeled as such?
pre = metrics.precision_score(y_test, y_pred)
print("Precision:", pre)

# Model Recall: what percentage of positive tuples are labelled as such?
rec = metrics.recall_score(y_test, y_pred)
print("Recall:", rec)

# Model F1
f1 = 2 * pre * rec / (pre + rec)
print("F1 score:", f1)
print(train_images.shape)

plt.scatter(train_images)
plt.show()