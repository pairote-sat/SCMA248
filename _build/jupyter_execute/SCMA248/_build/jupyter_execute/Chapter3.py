#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
pathlib.Path().resolve()


# In[2]:


import pandas as pd


# In[3]:


# pip install pyreadr


# In[4]:


import pyreadr


# In[5]:


# https://github.com/ofajardo/pyreadr#basic-usage--reading-files

result = pyreadr.read_r('/Users/Kaemyuijang/Documents/Github/pyreadr-master/test_data/basic/two.RData')

# done! let's see what we got
print(result.keys()) # let's check what objects we got
df1 = result["df1"] # extract the pandas data frame for object df1


# In[6]:


print(result.keys())


# In[7]:


import pyreadr

result = pyreadr.read_r('/Users/Kaemyuijang/Documents/Github/pyreadr-master/test_data/basic/one.Rds')

# done! let's see what we got
print(result.keys()) # let's check what objects we got: there is only None
df1 = result[None] # extract the pandas data frame for the only object available


# In[8]:


df1.values


# In[9]:


print('\n'.join(f'{m.__name__}=={m.__version__}' for m in globals().values() if getattr(m, '__version__', None)))


# In[10]:


result = pyreadr.read_r('~/Documents/Github/pyreadr-master/test_data/basic/myData.RData') # also works for Rds, rda


# In[11]:


print(result.keys())


# In[12]:


fre = result['freMTPL2freq']


# In[13]:


type(fre)

