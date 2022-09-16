#!/usr/bin/env python
# coding: utf-8

# In[1]:


from plotnine import *


# In[3]:


from plotnine.data import mtcars


# In[4]:


(ggplot(mtcars, aes('wt','mpg',color='factor(gear)')) +
     geom_point()
)


# In[8]:



from scipy import stats
import plotnine as pn
from plotnine.data import economics

loc, scale = economics['pce'].mean(), economics['pce'].std()
(pn.ggplot(economics[['pce','date']], pn.aes(x='pce'))
 +pn.geom_density()
 +pn.stat_function(color='red', fun=stats.norm.pdf, args=dict(loc=loc,scale=scale))
 )


# In[ ]:




