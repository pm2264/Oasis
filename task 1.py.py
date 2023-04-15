#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score


# In[3]:


df=pd.read_csv("E:\DATA\Iris.csv")
df.head()


# In[4]:


df.head(10)


# In[5]:


df.tail()


# In[6]:


df.tail()


# In[7]:


df.isnull().sum()


# In[8]:


df.dtypes


# In[9]:


data=df.groupby('Species')
data.head()


# In[10]:


df['Species'].unique()


# In[11]:


df.info()


# In[12]:


plt.boxplot(df['SepalLengthCm'])


# In[13]:


plt.boxplot(df['SepalWidthCm'])


# In[14]:


plt.boxplot(df['PetalLengthCm'])


# In[15]:


plt.boxplot(df['PetalWidthCm'])


# In[16]:


sns.heatmap(df.corr())


# In[17]:


df.drop('Id',axis=1,inplace=True)


# In[18]:


sp={'Iris-setosa':1,'Iris-versicolor':2,'Iris-virginica':3}
df.Species=[sp[i] for i in df.Species]

df


# In[19]:


X=df.iloc[:,0:4]
X


# In[20]:


y=df.iloc[:,4]
y


# In[22]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)


# In[23]:


model=LinearRegression()


# In[24]:


model.fit(X,y)


# In[25]:


model.score(X,y) #coef of prediction


# In[26]:


model.coef_


# In[27]:


model.intercept_


# In[28]:


y_pred=model.predict(X_test)


# In[29]:


print("Mean squared error: %.2f" % np.mean((y_pred - y_test) ** 2))


# In[ ]:




