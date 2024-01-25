#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[3]:


df=pd.read_csv(r'C:\Users\prakh\Downloads\Titanic-Dataset.csv')


# In[4]:


df.info


# In[5]:


df.head()


# In[6]:


df.describe()


# In[7]:


df.isnull().sum()#or df.isna().sum()


# In[8]:


#Data Cleaning 
df.drop('PassengerId',inplace=True,axis=1)


# In[9]:


df['Cabin'].count()/df.shape[0]


# In[10]:


df.drop('Cabin',inplace=True,axis=1)


# In[11]:


df.info()


# In[12]:


df['Age'].count()/df.shape[0]


# In[13]:


df.dropna(subset=['Age','Embarked'],inplace=True)


# In[14]:


df.info()


# In[15]:


df['Title']=df['Name'].str.extract(r'([A-Za-z]+\.)',expand=False)
df.drop('Name',inplace=True,axis=1)


# In[16]:


pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
df


# In[21]:


df.describe()


# In[ ]:




