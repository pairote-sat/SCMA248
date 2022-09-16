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


# In[46]:


print(result.keys())


# In[47]:


import pyreadr

result = pyreadr.read_r('/Users/Kaemyuijang/Documents/Github/pyreadr-master/test_data/basic/one.Rds')

# done! let's see what we got
print(result.keys()) # let's check what objects we got: there is only None
df1 = result[None] # extract the pandas data frame for the only object available


# In[48]:


df1.values


# In[49]:


print('\n'.join(f'{m.__name__}=={m.__version__}' for m in globals().values() if getattr(m, '__version__', None)))


# In[50]:


result = pyreadr.read_r('~/Documents/Github/pyreadr-master/test_data/basic/myData.RData') # also works for Rds, rda


# In[51]:


print(result.keys())


# In[52]:


fre = result['freMTPL2freq']


# In[1]:


type(fre)


# In[ ]:





# In[9]:


# Libraries needed for the tutorial

import pandas as pd
import requests
import io

url = "https://github.com/pairote-sat/SCMA248/blob/main/demo_df" # Make sure the url is the raw version of the file on GitHub
download = requests.get(url).content

df = pd.read_csv(io.StringIO(download.decode('utf-8')))

# Printing out the first 5 rows of the dataframe

print (df.head())


# In[11]:


import pandas as pd

url = 'https://raw.githubusercontent.com/pairote-sat/SCMA248/main/demo_df'
df = pd.read_csv(url, index_col=0)
print(df.head(5))


# In[ ]:


https://raw.githubusercontent.com/pairote-sat/SCMA248/main/demo_df

