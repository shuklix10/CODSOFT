#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import math


# In[5]:


df=pd.read_csv('C:/Users/prakh/Videos/IMDb Movies India.csv',encoding='latin1')
df.info()


# In[6]:


df.isnull().sum()


# In[7]:


df.head(10)


# In[8]:


#Data Cleaning
df.dropna(subset=df.columns[1:9],how='all',inplace=True)


# In[9]:


df.dropna(subset=['Name','Year'],how='all',inplace=True)


# In[10]:


df.drop_duplicates(['Name','Year'],keep='first',inplace=True)


# In[11]:


df.info()


# In[12]:


df.dropna(subset=['Year'],inplace=True)


# In[13]:


df['Year']=df['Year'].str.extract(r'([0-9].{0,3})',expand=False)


# In[14]:


df['Duration']=df['Duration'].str.extract(r'([0-9]+)',expand=False)


# In[15]:


def get_mode_with_default(x):
    mode_result = x.mode()
    if not mode_result.empty:
        return mode_result[0]
    else:
        return 'unknown'  

df['Actor 1']=df['Actor 1'].fillna(df.groupby('Year')['Actor 1'].transform(get_mode_with_default))
df['Actor 2']=df['Actor 2'].fillna(df.groupby('Year')['Actor 2'].transform(get_mode_with_default))
df['Actor 3']=df['Actor 3'].fillna(df.groupby('Year')['Actor 3'].transform(get_mode_with_default))


# In[16]:


df['Director']=df.groupby(['Year','Actor 1','Actor 2','Actor 3'])['Director'].transform(get_mode_with_default)


# In[17]:


df['Duration']=pd.to_numeric(df['Duration'])


# In[18]:


def get_mean_with_default(x):
    mean_result = x.mean()
    if not math.isnan(mean_result):        
            return round(mean_result)
    else:
        return 0
df['Duration']=df.groupby(['Year','Director','Actor 1','Actor 2','Actor 3'])['Duration'].transform(get_mean_with_default)


# In[19]:


df['Rating']=df.groupby(['Director','Actor 1'])['Rating'].transform(lambda x:x.mean())
df['Rating']=df.groupby(['Director','Actor 2'])['Rating'].transform(lambda x:x.mean())
df['Rating']=df.groupby(['Director','Actor 3'])['Rating'].transform(lambda x:x.mean())
df['Rating']=df.groupby(['Year','Director'])['Rating'].transform(lambda x:x.mean())
df['Rating']=df.groupby('Year')['Rating'].transform(lambda x:x.mean())
df['Year']=pd.to_numeric(df['Year'])


# In[20]:


df['Votes']=df['Votes'].str.extract(r'([0-9]+)',expand=False)
df['Votes']=pd.to_numeric(df['Votes'])


# In[21]:


df['Votes']=df.groupby(['Year','Rating'])['Votes'].transform(lambda x:x.mean())


# In[22]:


df['Votes']=df.groupby('Year')['Votes'].transform(lambda x:x.mean())


# In[23]:


df.info()


# In[28]:


#EDA
rating_sum=df.groupby('Year')['Rating'].sum().reset_index()
plt.figure(figsize=(12,6))
sns.lineplot(x='Year',y='Rating',data=rating_sum)
sns.scatterplot(x='Year',y='Rating',data=rating_sum,color='r')
plt.yticks(np.arange(0,3000,400))
plt.xticks(np.arange(1920,2025,5))
plt.ylabel('Ratings')
plt.xlabel('Years')
plt.title('Ratings Per Years')
plt.xticks(rotation=90)
plt.grid(True)
plt.show()


# In[29]:


#year of movie  with best rating avverage 
rating_avg=df.groupby('Year')['Rating'].mean().reset_index()

plt.figure(figsize=(20,6))
sns.lineplot(x='Year',y='Rating',data=rating_avg)
sns.scatterplot(x='Year',y='Rating',data=rating_avg,color='r')
plt.yticks(np.arange(4,8,0.5))
plt.xticks(np.arange(1920,2025,5))
plt.ylabel('Average Ratings')
plt.xlabel('Years')
plt.title('Average Ratings Per Years')
plt.xticks(rotation=90)
plt.grid(True)
plt.show()


# In[30]:


#Top 20 Directors by Frequency of Movies

top_20=df.groupby('Director')['Name'].count()[0:20]
sns.barplot(x=top_20.index,y=top_20.values,data=df,palette='viridis')
plt.xticks(rotation=90)
plt.ylabel('Frequency Of Movies')
plt.xlabel('Director')
plt.show()


# In[31]:


#Does length of movie have any impact with the rating
corr_leng_rat=df['Duration'].corr(df['Rating'])
print(f"Correlation Of Duration And Rating is {corr_leng_rat}")
#show there is no impact of duration on rating

plt.figure(figsize=(8,6))
sns.scatterplot(x='Duration',y='Rating',data=df)
plt.xlabel('Duration')
plt.ylabel('Rating')
plt.title('Duration Vs Rating')
plt.yticks(np.arange(4,8,0.5))
plt.show()


# In[32]:


#Top 10 movies according to rating per year and overall.
overall=df.nlargest(10,'Rating')
overall=overall.reset_index(drop=True)
print("Top 10 Movies Overall:")
overall


# In[35]:


#Number of popular movies released each year.
rat_bool=df['Rating']>=6
vot_bool=df['Votes']>110
pop_df=df[vot_bool & rat_bool]
pop_df


# In[36]:


# ML
df.dropna(inplace=True)
df.isnull().sum()


# In[37]:


fig,ax=plt.subplots(nrows=2,ncols=2,figsize=(10,6))

sns.boxplot(data=df,y='Rating',ax=ax[0][0])
ax[0][0].set_title('Ratings')
ax[0][0].set_xlabel('Ratings')

sns.boxplot(data=df,y='Duration',ax=ax[0][1])
ax[0][1].set_title('Duration')
ax[0][1].set_xlabel('Duration')

sns.boxplot(data=df,y='Votes',ax=ax[1][0])
ax[1][0].set_title('Votes')
ax[1][0].set_xlabel('Votes')

sns.boxplot(data=df,y='Year',ax=ax[1][1])
ax[1][1].set_title('Years')
ax[1][1].set_xlabel('Years')

plt.tight_layout()

plt.show()


# In[38]:


def out(df,col,dis):
    q1=df[col].quantile(0.25)
    q3=df[col].quantile(0.75)
    iqr=q3-q1
    lower=q1-(iqr*dis)
    upper=q3+(iqr*dis)
    return lower,upper


# In[39]:


votes_low,votes_up=out(df,'Votes',1.5)


# In[40]:


vote_out_count=(df['Votes'] > votes_up) | (df['Votes'] < votes_low)


# In[41]:


df['Votes'][vote_out_count].count()


# In[42]:


df=df[(df['Votes']>votes_low) & (df['Votes']<votes_up)]


# In[43]:


year_low,year_upper=out(df,'Year',1.5)


# In[44]:


year_out_count=(df['Year']>year_upper) | (df['Year']<year_low)


# In[45]:


df['Year'][year_out_count].count()


# In[47]:


#Appling ML 
from sklearn.preprocessing import LabelEncoder
LB=LabelEncoder()
df['Name']=LB.fit_transform(df['Name'])
df['Genre']=LB.fit_transform(df['Genre'])
df['Director']=LB.fit_transform(df['Director'])
df['Actor 1']=LB.fit_transform(df['Actor 1'])
df['Actor 2']=LB.fit_transform(df['Actor 2'])
df['Actor 3']=LB.fit_transform(df['Actor 3'])


# In[48]:


from sklearn.linear_model import LinearRegression
LR=LinearRegression()


# In[49]:


from sklearn.model_selection import train_test_split
x=df.drop('Rating',axis=1)
y=df['Rating']


# In[51]:


train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=42)


# In[52]:


LR.fit(train_x,train_y)
pre_test=LR.predict(test_x)


# In[53]:


pre_test


# In[54]:


pre_train=LR.predict(train_x)


# In[55]:


from sklearn.metrics import r2_score
score_test=r2_score(test_y,pre_test)
score_train=r2_score(train_y,pre_train)
print("print r2_score",score_test)
print('print r2_score',score_train)


# In[56]:


from sklearn.linear_model import Ridge
RL=Ridge(alpha=10.0)
RL.fit(train_x,train_y)
RL_pre_test=RL.predict(test_x)
RL_pre_train=RL.predict(train_x)
r2_RL_test=r2_score(test_y,RL_pre_test)
r2_RL_train=r2_score(train_y,RL_pre_train)
print("print r2_score",r2_RL_test)
print('print r2_score',r2_RL_train)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(test_y, RL_pre_test)
print(mse)


# In[57]:


from sklearn.model_selection import GridSearchCV

param={'alpha':[0.01, 0.1, 1.0, 10.0]}
grid=GridSearchCV(estimator=RL,param_grid=param,cv=5)
grid.fit(train_x,train_y)

print(grid.best_params_,grid.best_estimator_)


# In[58]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
kf=KFold(n_splits=10,random_state=42,shuffle=True)
cv=cross_val_score(RL,x,y,cv=kf,n_jobs=-1)
print('Accuracy : ',cv.mean()*100)


# In[59]:


from sklearn.model_selection import RepeatedKFold
rfk=RepeatedKFold(n_splits=10,random_state=42,n_repeats=5)
cv1=cross_val_score(RL,x,y,cv=rfk,n_jobs=-1)
print('Accuracy : ',cv1.mean()*100)


# In[ ]:




