#!/usr/bin/env python
# coding: utf-8

# # Resampling, Shifting, and Windowing
# 
# Ref: VanderPlas, Jacob T - Python data science handbook_ essential tools for working with data-O'Reilly Media (2017)

# In[1]:


pip install pandas-datareader


# In[2]:


from pandas_datareader import data


# In[3]:


# https://github.com/jakevdp/PythonDataScienceHandbook/issues/94

import pandas_datareader.data as web
from datetime import datetime

goog = data.DataReader('F', start='2004', end='2016',data_source='yahoo')

goog.head()


# In[4]:


goog = goog['Close']


# In[5]:


goog


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn; seaborn.set()


# In[16]:


goog.plot();


# In[27]:


print(goog.head(5))
print(goog.shift(1).fillna(0).head(3))


# In[23]:


rolling = goog.rolling(5, center=True)


# In[26]:


rolling.mean().head()


# In[35]:





# In[2]:


for i in range(5):
    print(i)


# In[ ]:




