#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
from plotnine import *
from plotnine.data import mpg

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


(ggplot(mpg)
 + aes(x='manufacturer')
 + geom_bar(size=20)
 + coord_flip()
 + labs(y='Count', x='Manufacturer', title='Number of Cars by Make')
)


# In[3]:


# Determine order and create a categorical type
# Note that value_counts() is already sorted
manufacturer_list = mpg['manufacturer'].value_counts().index.tolist()
manufacturer_cat = pd.Categorical(mpg['manufacturer'], categories=manufacturer_list)

# assign to a new column in the DataFrame
mpg = mpg.assign(manufacturer_cat = manufacturer_cat)

(ggplot(mpg)
 + aes(x='manufacturer_cat')
 + geom_bar(size=20)
 + coord_flip()
 + labs(y='Count', x='Manufacturer', title='Number of Cars by Make')
)


# In[4]:


# Determine order and create a categorical type
# Note that value_counts() is already sorted
manufacturer_list = mpg['manufacturer'].value_counts().index.tolist()

(ggplot(mpg)
 + aes(x='manufacturer_cat')
 + geom_bar(size=20)
 + scale_x_discrete(limits=manufacturer_list)
 + coord_flip()
 + labs(y='Count', x='Manufacturer', title='Number of Cars by Make')
)


# In[5]:


# Determine order and create a categorical type
# Note that value_counts() is already sorted
manufacturer_list = mpg['manufacturer'].value_counts().index.tolist()[::-1]

(ggplot(mpg)
 + aes(x='manufacturer_cat')
 + geom_bar(size=20)
 + scale_x_discrete(limits=manufacturer_list)
 + coord_flip()
 + labs(y='Count', x='Manufacturer', title='Number of Cars by Make')
)


# In[8]:


mpg.head(20)


# In[12]:


mpg['manufacturer'].value_counts().index


# In[ ]:




