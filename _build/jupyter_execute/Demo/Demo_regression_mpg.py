#!/usr/bin/env python
# coding: utf-8

# In[1]:


# sm.graphics.plot_regress_exog(model1,"x")


# #### Making Predictions based on the Regression Results
# 
# 
# 
# 
# 
# 
# 

# In[2]:


import pandas as pd
import numpy as np

from plotnine import *
from plotnine.data import *

get_ipython().run_line_magic('matplotlib', 'inline')


(
    ggplot(mpg, aes(x='displ', y='hwy'))
    + geom_point()
    + geom_smooth(method='lm')
    + labs(x='displacement', y='horsepower')
)


# In[22]:


import plotnine as p9
from scipy import stats
from plotnine.data import mtcars as df

#calculate best fit line
slope, intercept, r_value, p_value, std_err = stats.linregress(df['wt'],df['mpg'])
df['fit']=df.wt*slope+intercept
#format text 
txt= 'y = {:4.2e} x + {:4.2E};   R^2= {:2.2f}'.format(slope, intercept, r_value*r_value)
#create plot. The 'factor' is a nice trick to force a discrete color scale
plot=(p9.ggplot(data=df, mapping= p9.aes('wt','mpg', color = 'factor(gear)'))
    + p9.geom_point(p9.aes())
    + p9.xlab('Wt')+ p9.ylab(r'MPG')
    + p9.geom_line(p9.aes(x='wt', y='fit'), color='black')
    + p9.annotate('text', x= 3, y = 35, label = txt))
#for some reason, I have to print my plot 
print(plot)


# In[23]:


dir(model1)

