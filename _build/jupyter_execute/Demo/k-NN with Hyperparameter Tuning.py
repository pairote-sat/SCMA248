#!/usr/bin/env python
# coding: utf-8

# https://www.kaggle.com/code/arunimsamudra/k-nn-with-hyperparameter-tuning/notebook

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns # for data visualiztions


# In[25]:


path = 'https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv'

df = pd.read_csv(path)


# In[26]:


df.columns


# In[27]:


# Separate the dependent and independent features
X = df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
y = df['variety']


# In[28]:


# Standardize the data and then concatenate it with y
data = X
data_std = (data - data.mean())/(data.max() - data.min())
data = pd.concat([data_std,y], axis=1)


# In[29]:


# reshape the dataframe using melt()
data = pd.melt(data, id_vars = 'variety', var_name = 'features',value_name = 'value')
#data.rename(columns={"variety": "Species"})


# In[30]:


data


# In[31]:


# swarmplot for analysing the different attributes
plt.figure(figsize = (6,6))
sns.swarmplot(x = 'features', y = 'value', hue = 'variety', data = data)
plt.show()


# #### Feature Selection
# 

# In[32]:


# split the dataset into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
print('Training set shape: ', X_train.shape, y_train.shape)
print('Testing set shape: ', X_test.shape, y_test.shape)


# In[33]:


from sklearn.feature_selection import chi2, SelectKBest, f_classif


# In[34]:


# Get the two best(k = 2) features using the SelectKBest method
ft = SelectKBest(chi2, k = 2).fit(X_train, y_train)
print('Score: ', ft.scores_)
print('Columns: ', X_train.columns)


# In[ ]:




