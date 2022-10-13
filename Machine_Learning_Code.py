#IMPORT PACKAGES
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import lightgbm as lgb

from multiprocessing.sharedctypes import Value
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

from multiprocessing.sharedctypes import Value
from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_squared_error
from catboost import CatBoostClassifier, Pool


# import data 
data = pd.read_csv('/Users/kuang/Dropbox (MIT)/SarahOh/dataset.csv')
X = data[['ethnicity', 'race','maternal_age','complications', 'type', 'prenatal_care', 'first_frequency', 'first_amount', 'second_frequency', 'second_amount', 'third_frequency', 'third_amount']]
y = data['fas_f']

# create categorical variables 
X = pd.get_dummies(X, columns=['race', 'type', 'ethnicity'], drop_first=True)

#split the dataset into training (70%) and testing (30%) sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
classes = ['no FAS', 'FAS']


# 1) logistic regression
#FIT LOGISTIC REGRESSION MODEL
#instantiate the model
log_regression = LogisticRegression(max_iter=1000)
#fit the model using the training data
log_regression.fit(X_train,y_train)

#use model to make predictions on test data
y_pred = log_regression.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("Confusion Matrix:",metrics.confusion_matrix(y_test, y_pred))
print("Classification Report:", metrics.classification_report(y_test, y_pred))

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
# plt.show()


# 2) lightgbm
# build the lightgbm model
clf = lgb.LGBMClassifier(learning_rate=0.1, max_depth=3)
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("Confusion Matrix:",metrics.confusion_matrix(y_test, y_pred))
print("Classification Report:", metrics.classification_report(y_test, y_pred))

cm2 = confusion_matrix(y_test, y_pred)
#Confusion matrix, Accuracy, sensitivity and specificity
sensitivity1 = cm2[0,0]/(cm2[0,0]+cm2[0,1])
print('Sensitivity : ', sensitivity1 )
specificity1 = cm2[1,1]/(cm2[1,0]+cm2[1,1])
print('Specificity : ', specificity1)

#plot ROC curve
y_pred_proba = clf.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.legend(loc=4)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
# plt.show()


# 3) XGBoost 
# create model instance
bst = XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.1, objective='binary:logistic')
# fit model
bst.fit(X_train, y_train)
# make predictions
y_pred = bst.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("Confusion Matrix:",metrics.confusion_matrix(y_test, y_pred))
print("Classification Report:", metrics.classification_report(y_test, y_pred))

cm3 = confusion_matrix(y_test, y_pred)
#Confusion matrix, Accuracy, sensitivity and specificity
sensitivity1 = cm3[0,0]/(cm3[0,0]+cm3[0,1])
print('Sensitivity : ', sensitivity1 )
specificity1 = cm3[1,1]/(cm3[1,0]+cm3[1,1])
print('Specificity : ', specificity1)

#plot ROC curve
y_pred_proba = bst.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.legend(loc=4)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
# plt.show()


# 4) Catboost
# build the Catboost model
model = CatBoostClassifier(iterations=500, depth=3, learning_rate=0.1,loss_function='Logloss',verbose=False)
model.fit(X_train, y_train)
y_pred=model.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("Confusion Matrix:",metrics.confusion_matrix(y_test, y_pred))
print("Classification Report:", metrics.classification_report(y_test, y_pred))

cm4 = confusion_matrix(y_test, y_pred)
#Confusion matrix, Accuracy, sensitivity and specificity
sensitivity1 = cm4[0,0]/(cm4[0,0]+cm4[0,1])
print('Sensitivity : ', sensitivity1 )
specificity1 = cm4[1,1]/(cm4[1,0]+cm4[1,1])
print('Specificity : ', specificity1)

#plot ROC curve
y_pred_proba = model.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.legend(loc=4)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
# plt.show()


# plot confusion matrices
disp = ConfusionMatrixDisplay(confusion_matrix=cm1,display_labels=classes)
disp.plot()
plt.title('Logistic Regression')
# plt.show()

disp = ConfusionMatrixDisplay(confusion_matrix=cm2,display_labels=classes)
disp.plot()
plt.title('lightgbm')
# plt.show()

disp = ConfusionMatrixDisplay(confusion_matrix=cm3,display_labels=classes)
disp.plot()
plt.title('XGBoost')
# plt.show()

disp = ConfusionMatrixDisplay(confusion_matrix=cm4,display_labels=classes)
disp.plot()
plt.title('CatBoost')
plt.show()