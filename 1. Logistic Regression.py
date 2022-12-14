#IMPORT PACKAGES
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('/Users/sarahoh/Desktop/py_scripts/hello/220901/dataset.csv')

#FIT LOGISTIC REGRESSION MODEL
X = data[['ethnicity', 'race','maternal_age','complications', 'type', 'prenatal_care', 'first_frequency', 'first_amount', 'second_frequency', 'second_amount', 'third_frequency', 'third_amount']]
y = data['fas_f']

#split the dataset into training (70%) and testing (30%) sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

#instantiate the model
log_regression = LogisticRegression(max_iter=100)
#fit the model using the training data
log_regression.fit(X_train,y_train)

#use model to make predictions on test data
y_pred = log_regression.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("Confusion Matrix:",metrics.confusion_matrix(y_test, y_pred))
print("Classification Report:", metrics.classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred)
#Confusion matrix, Accuracy, sensitivity and specificity
sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
print('Sensitivity : ', sensitivity1 )
specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
print('Specificity : ', specificity1)
pos_pred_val = cm1[0,0]/(cm1[0,0]+cm1[0,1])

#plot ROC curve
y_pred_proba = log_regression.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.legend(loc=4)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

