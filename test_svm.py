# -*- coding: utf-8 -*-
"""
Created on Tue May  5 13:06:37 2020

@author: Kamal Boulahya
"""
#Import scikit-learn dataset library
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from tqdm import tqdm
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform
import warnings
#--------------------------------------------------------#
#                       Data
#--------------------------------------------------------#
train_images = np.load(r'train_images_1k.npy')
train_label = np.load(r'train_label_1k.npy')
train_images = train_images.reshape(1000, 32*32*3)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(train_images,train_label, test_size=0.1,random_state=109) # 90% training and 10% test



scaler1=StandardScaler()
scaler1.fit(X_test)
X_test_scaled=scaler1.transform(X_test)
#PCA
pca1=PCA(n_components=2)
X_test_scaled_reduced=pca1.fit_transform(X_test_scaled)
X_pca=pca1.transform(X_test_scaled)
#Variance: vérifier les dimensions importantes lors de l'estimation
feat_var =np.var(X_pca,axis=0)
feat_var_rat= feat_var/(np.sum(feat_var))
print(feat_var_rat)

#--------------------------------------------------------#
#       Cross-validation method: GridSearchCV
#--------------------------------------------------------#

# tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.001,0.005,0.01,0.05,0.07,0.1],
#                       'C': [1, 10, 100, 1000]},
#                     {'kernel': ['sigmoid'],'gamma': [0.001,0.005,0.01,0.05,0.07,0.1],
#                       'C': [1, 10, 100, 1000]}]

# scores = ['recall','f1']

# for score in scores:
#     print("# Tuning hyper-parameters for %s" % score)
#     print()

#     clf = GridSearchCV(SVC(), tuned_parameters, scoring='%s' %score)
#     clf.fit(X_train, y_train)

#     print("Best parameters set found on development set:")
#     print()
#     print(clf.best_params_)
#     print()
#     print("Grid scores on development set:")
#     print()
#     means = clf.cv_results_['mean_test_score']
#     stds = clf.cv_results_['std_test_score']
#     for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#         print("%0.3f (+/-%0.03f) for %r"
#               % (mean, std * 2, params))
#     print()

#     print("Detailed classification report:")
#     y_true, y_pred = y_test, clf.predict(X_test)
#     # Model Accuracy: how often is the classifier correct?
#     acc = metrics.accuracy_score(y_true, y_pred)
#     print("Accuracy:", acc)
#     # Model Precision: what percentage of positive tuples are labeled as such?
#     pre = metrics.precision_score(y_true, y_pred)
#     print("Precision:", pre)
#     # Model Recall: what percentage of positive tuples are labelled as such?
#     rec = metrics.recall_score(y_true, y_pred)
#     print("Recall:", rec)
#     # Model F1
#     f1 = 2 * pre * rec / (pre + rec)
#     print("F1 score:", f1)


#--------------------------------------------------------#
#       Cross-validation method: RandomSearchCV
#--------------------------------------------------------#
# print('Random Search')
# clf = RandomizedSearchCV(SVC(kernel='rbf'), dict(C=uniform(loc=0, scale=100),gamma=uniform(loc=0.0001, scale=100)))
# search = clf.fit(X_train,y_train)
# print(search.best_params_)    

#--------------------------------------------------------#
#    Cross-validation method: Pipeline + GridSearhCV
#--------------------------------------------------------#
# #Pipeline
# pipe_steps = [('scaler', StandardScaler()),('pca',PCA()),('SupVM',SVC(cache_size=1000))]

# check_params = {
#     'pca__n_components': [2],
#     'SupVM__C':[0.1,0.5,1,10,30,40,50,75,100,500,1000],
#     'SupVM__gamma': [0.001,0.005,0.01,0.05,0.07,0.1,0.5,1,5,10,50],
#     'SupVM__kernel':['rbf']
#     },{
#     'pca__n_components': [2],
#     'SupVM__C':[0.1,0.5,1,10,30,40,50,75,100,500,1000],
#     'SupVM__gamma': [0.001,0.005,0.01,0.05,0.07,0.1,0.5,1,5,10,50],
#     'SupVM__kernel':['sigmoid']
#     }
# pipeline=Pipeline(pipe_steps)

# #Training
# print("Start Fitting Training Data")
# for cv in tqdm(range(4,6)):
#     create_grid = GridSearchCV(pipeline, param_grid=check_params,cv=cv,scoring='f1')
#     create_grid.fit(X_train,y_train)
#     print("\n Score for %d fold CV := %3.2f" %(cv, create_grid.score(X_test,y_test)))
#     print('Best fit parameters from training data:')
#     print(create_grid.best_params_)

#--------------------------------------------------------#
#                     SVM Classifier
#--------------------------------------------------------#
warnings.filterwarnings("ignore")
#Create a svm Classifier
svm_model = SVC(kernel='sigmoid', C=1000, gamma=0.07) # Linear Kernel
svm_model2 = SVC(kernel='rbf', C=80, gamma=40) # Linear Kernel
#Train the model using the training sets
clf=svm_model.fit(X_train, y_train)
clf2=svm_model2.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)
y_pred2 = clf2.predict(X_test)


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

# Model Accuracy: how often is the classifier correct?
acc2 = metrics.accuracy_score(y_test, y_pred2)
print("Accuracy:", acc)
# Model Precision: what percentage of positive tuples are labeled as such?
pre2 = metrics.precision_score(y_test, y_pred2)
print("Precision:", pre)
# Model Recall: what percentage of positive tuples are labelled as such?
rec2 = metrics.recall_score(y_test, y_pred2)
print("Recall:", rec)
# Model F1
f1_2 = 2 * pre * rec / (pre + rec)
print("F1 score:", f1_2)

