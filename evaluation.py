# -*- coding: utf-8 -*-

#Prediction on test set
from sklearn.metrics import roc_curve,auc,precision_recall_curve,average_precision_score, classification_report, confusion_matrix
label_pred_keras = model.predict(test_images,verbose =1)[:,1]


#ROC for False positive and True Positives from pred + AUC
fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_label, label_pred_keras)
auc_keras = auc(fpr_keras, tpr_keras)
#ROC Curve
plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='SVM (AUC = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

#Precision/Recall 
lr_precision, lr_recall, thresholds = precision_recall_curve(test_label, label_pred_keras)
#Curve
plt.figure()
plt.plot(lr_recall, lr_precision, linestyle='--', label='SVM')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
#Average Precision on test
print("Average Precision Score: ", average_precision_score(test_label, label_pred_keras))
#fbeta_score(y_true, y_pred) renvoie le score f1 avec un poids important soit à la précision soit au recall


""" ERROR BINARY AND CONTINOUS
#Confusion matrix
labels = ['Cancer not detected', 'Cancer detected']
cm = confusion_matrix(test_label, label_pred_keras, labels)
print(cm)
#Board
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the Keras classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

#Recap of precision, recall, f1, support
target_names = ['class 0', 'class 1']
report = classification_report(test_label, label_pred_kera,target_names=target_names)
"""