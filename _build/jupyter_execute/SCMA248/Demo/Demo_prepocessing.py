#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data = {'Country': ['Belgium', 'India', 'Brazil'], 'Capital': ['Brussels', 'New Delhi', 'Brasília'], 'Population': [11190846, 1303171035, 207847528]}


# In[3]:


type(data)


# In[4]:


type(data['Population'][0])


# In[5]:


df = pd.DataFrame(data, columns=['Country', 'Capital', 'Population'])


# In[6]:


type(df)


# In[7]:


df.columns


# In[8]:


df['Capital'][0:]


# In[9]:


df['Capital'][0:]


# In[10]:


df['Capital'][0]


# Data Selection with Pandas

# In[11]:


df['Capital'][2]


# In[12]:


df[2]['Capital']


# In[47]:


df[2:3]['Capital']


# In[48]:


df['Capital'][2:3]


# In[52]:


df.loc[0:1,'Capital']


# In[53]:


df.iloc[0:1,0]


# In[ ]:





# In[54]:





# Masking p.79
# 

# In[57]:


iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None, 
                names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])

type(iris)


# In[59]:


iris.columns


# In[65]:


iris[iris['sepal_length'] > 6]['sepal_length']


# In[71]:


iris.groupby(['class']).mean()


# In[72]:


iris.groupby(['sepal_length']).mean()


# In[78]:


iris.columns


# In[81]:


iris.sort_index(ascending=False).head()


# In[82]:


iris.sort_index(axis=1).head()


# In[73]:


iris.sort_values(by = 'sepal_length').head()


# Counting

# In[85]:


import numpy as np


# In[87]:


iris.apply(np.count_nonzero, axis =0)


# In[89]:


iris.apply(np.count_nonzero, axis =1)


# In[90]:


x = lambda a : a + 10
print(x(5))


# In[91]:


x = lambda a, b : a * b
print(x(5, 6))


# In[92]:


def myfunc(n):
  return lambda a : a * n

mydoubler = myfunc(2)

print(mydoubler(11))


# In[5]:


import pandas as pd

df = pd.DataFrame( [[4, 7, 10], [5, 7, 11],[6, 9, 12]],index=[1, 2, 3], columns=['a', 'b', 'c'])


# In[6]:


df


# In[7]:


df['b'].value_counts()


# In[ ]:




