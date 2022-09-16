#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

from plotnine import *
from plotnine.data import *

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


mpg.head()


# To change the plot order of the rows or columns in the facet grid, reorder the levels of the faceting variable in the data.

# In[3]:


# re-order categories
mpg['drv'] = mpg['drv'].cat.reorder_categories(['f', 'r','4'])


# In[5]:


# facet plot with reorded drv category
(
    ggplot(mpg, aes(x='displ', y='hwy'))
    + geom_point()
    + facet_grid('drv ~ cyl')
    + labs(x='displacement', y='horsepower')
)


# In[6]:


mpg.columns


# In[14]:


mpg.pipe(lambda x: x.assign(new_column = x.fl))


# In[ ]:


gapminder.filter(['country','dollars_per_day','continent'])

