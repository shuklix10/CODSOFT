#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.linear_model import LogisticRegression


# In[2]:


credit_card_data = pd.read_csv(r'C:\Users\prakh\Videos\creditcard.csv')


# In[3]:


credit_card_data 


# In[4]:


ps = credit_card_data


# In[5]:


ps.info()


# In[6]:


ps.isnull().sum()


# In[7]:


ps['Class'].value_counts()


# In[8]:


print((ps.groupby('Class')['Class'].count()/ps['Class'].count())*100)
((ps.groupby('Class')['Class'].count()/ps['Class'].count())*100).plot.pie()


# In[9]:


classes = ps['Class'].value_counts()
normal_value = round(classes[0]/ps['Class'].count()*100,2)
fraud_values = round(classes[1]/ps['Class'].count()*100,2)
print(normal_value)
print(fraud_values)


# In[10]:


corr = ps.corr()
corr


# In[11]:


plt.figure(figsize=(27,19))
sns.heatmap(corr, cmap = 'spring', annot= True )
plt.show()


# In[12]:


legit = ps[ps.Class == 0]


# In[13]:


fraud = ps[ps.Class==1]


# In[14]:


legit.Amount.describe()


# In[15]:


fraud.Amount.describe()


# In[16]:


ps.groupby('Class').describe()


# In[17]:


ps.groupby('Class').mean()


# In[18]:


normal_sample = legit.sample(n=495)


# In[21]:


ap= pd.concat([normal_sample, fraud], axis = 0)


# In[22]:


ap


# In[23]:


ap['Class'].value_counts()


# In[24]:


ap.groupby('Class').mean() 


# In[25]:


delta_time = pd.to_timedelta(ap['Time'], unit = 's')


# In[26]:


ap['time_hour']=(delta_time.dt.components.hours).astype(int)


# In[27]:


ap.drop(columns='Time', axis=1, inplace = True)


# In[28]:


ap


# In[29]:


x = ap.drop('Class', axis=1)


# In[30]:


y = ap['Class']


# In[31]:


x.shape


# In[32]:


y.shape


# In[33]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state = 3, stratify = y)


# In[34]:


cols = list(x.columns.values)


# In[35]:


normal_entries = ap.Class==0
fraud_entries = ap.Class==1

plt.figure(figsize=(20,70))
for n, col in enumerate(cols):
    plt.subplot(10,3,n+1)
    sns.histplot(x[col][normal_entries], color='blue', kde = True, stat = 'density')
    sns.histplot(x[col][fraud_entries], color='red', kde = True, stat = 'density')
    plt.title(col, fontsize=17)
plt.show()


# In[38]:


model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_train)
pred_test = model.predict(x_test)


# In[39]:


from sklearn.metrics import confusion_matrix
def Plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test,pred_test)
    plt.clf()
    plt.show()


# In[40]:


acc_score= round(accuracy_score(y_pred, y_train)*100,2)


# In[41]:


print('the accuracy score for training data of our model is :', acc_score)


# In[42]:


y_pred = model.predict(x_test)
acc_score = round(accuracy_score(y_pred, y_test)*100,2)


# In[43]:


print('the accuracy score of our model is :', acc_score)


# In[44]:


from sklearn import metrics


# In[45]:


score = round(model.score(x_test, y_test)*100,2)
print('score of our model is :', score)


# In[46]:


class_report = classification_report(y_pred, y_test)
print('classification report of our model: ', class_report)


# In[ ]:




