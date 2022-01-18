#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.display import HTML
import pandas as pd, numpy as np
out_df = pd.DataFrame({'hey': np.random.uniform(0,10, size = 5)})
HTML(out_df.to_html())


# In[ ]:




