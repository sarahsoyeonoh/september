#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd


# In[46]:


data = pd.read_csv('drinkers2.csv')


# In[47]:


type(data)


# In[136]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[49]:


data.shape


# In[54]:


data.head()


# In[55]:


data.isnull().sum()


# In[56]:


from sklearn.utils import shuffle


# In[57]:


data = shuffle(data, random_state = 42)

div = int(data.shape[0]/4)

train = data.loc[:3*div+1,:]
test = data.loc[3*div+1:]


# In[58]:


train.head()


# In[59]:


test.head()


# In[60]:


x = data.drop(['fas_f'], axis=1)
y = data['fas_f']
x.shape, y.shape


# In[61]:


from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(x,y, random_state = 56)


# In[62]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[63]:


cols = train_x.columns
cols


# In[66]:


train_x_scaled = scaler.fit_transform(train_x)
train_x_scaled = pd.DataFrame(train_x_scaled, columns=cols)
train_x_scaled.head()


# In[67]:


test_x_scaled = scaler.fit_transform(test_x)
test_x_scaled = pd.DataFrame(train_x_scaled, columns=cols)
test_x_scaled.head()


# In[68]:


from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.metrics import f1_score


# In[99]:


logreg = LogReg(max_iter=1000)
logreg.fit(train_x, train_y)


# In[137]:


train_predict = logreg.predict(train_x)


# In[101]:


k = f1_score(train_predict, train_y)
print('Training f1_score', k)


# In[102]:


test_predict = logreg.predict(test_x)
k = f1_score(test_predict, test_y)
print('Test f1_score', k)


# In[138]:


train_predict = logreg.predict_proba(train_x)


# In[139]:


train_preds = train_predict[:,1]


# In[107]:


for i in range(0, len(train_preds)):
    if(train_preds[i]>0.55):
        train_preds[i] = 1
    else:
        train_preds[i]=0


# In[110]:


k = f1_score(train_preds, train_y)
print('Training f1_score', k)


# In[111]:


from sklearn.metrics import confusion_matrix
cf= confusion_matrix(test_y, test_predict)
print(cf)


# In[112]:


from sklearn.metrics import classification_report as rep
print(rep(test_y, test_predict))


# In[124]:


import matplotlib.pyplot as plt
from sklearn import metrics


# In[125]:


#Confusion matrix, Accuracy, sensitivity and specificity
sensitivity1 = cf[0,0]/(cf[0,0]+cf[0,1])
print('Sensitivity : ', sensitivity1 )
specificity1 = cf[1,1]/(cf[1,0]+cf[1,1])
print('Specificity : ', specificity1)
pos_pred_val = cf[0,0]/(cf[0,0]+cf[0,1])


# In[127]:


#plot ROC curve
y_pred_proba = logreg.predict_proba(test_x)[::,1]
fpr, tpr, _ = metrics.roc_curve(test_y, y_pred_proba)
auc = metrics.roc_auc_score(test_y, y_pred_proba)
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.legend(loc=4)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:




