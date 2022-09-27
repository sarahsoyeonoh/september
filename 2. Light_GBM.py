from multiprocessing.sharedctypes import Value
from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import lightgbm as lgb
import numpy as np

data = pd.read_csv('/Users/sarahoh/Desktop/py_scripts/hello/220901/dataset.csv')

X = data[['ethnicity', 'race','maternal_age','complications', 'type', 'prenatal_care',
 'first_frequency', 'first_amount', 'second_frequency', 'second_amount', 'third_frequency', 'third_amount']]
y = data['fas_f']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

# build the lightgbm model
import lightgbm as lgb
clf = lgb.LGBMClassifier()
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)

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

#plot ROC curve
y_pred_proba = clf.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.legend(loc=4)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

