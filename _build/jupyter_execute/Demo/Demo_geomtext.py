#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import datetime

from plotnine import *



df = pd.DataFrame({
    'date':pd.date_range(start='1/1/1996', periods=4*25, freq='Q'),
    'small': pd.Series([0.035]).repeat(4*25) ,
    'large': pd.Series([0.09]).repeat(4*25),
})


# In[24]:


(ggplot()
    + geom_step(df, aes(x='date', y='small'))
    + geom_step(df, aes(x='date', y='large'))
    + scale_y_continuous(labels=lambda l: ["%d%%" % (v * 100) for v in l])
    + labs(x=None, y=None) 
    + geom_text(aes(x=pd.Timestamp('2000-01-01'), y = 0.0275), label = 'small')
)


# In[14]:


(ggplot(df)
 ...
 # + geom_text(aes(x=pd.Timestamp('2000-01-01'), y = 0.0275, label = '"small"'))
 # + geom_text(aes(x=pd.Timestamp('2000-01-01'), y = 0.0275), label = 'small')
 + annotate('text', x=pd.Timestamp('2000-01-01'), y = 0.0275, label='small')
)


# In[ ]:




