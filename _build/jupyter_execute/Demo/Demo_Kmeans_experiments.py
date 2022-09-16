#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# ## Experiments

# In[1]:


features, true_labels = make_blobs(
    n_samples=400,
    centers=4,
    cluster_std=0.5,
    random_state=1
)

df = pd.DataFrame(features, columns=["x1", "x2"])

scaler = StandardScaler()

df_scaled = df
features = [['x1','x2']]
for feature in features:
    df_scaled[feature] = scaler.fit_transform(df_scaled[feature])
df_scaled.plot.scatter("x1", "x2")


    
df_melted = pd.melt(df_scaled[['x1','x2']], var_name = 'features',value_name = 'value')    

(
    ggplot(df_melted) + aes('value') + geom_histogram(aes(y=after_stat('density')), color = 'lightskyblue', fill = 'lightskyblue', bins = 15)
    + geom_density(aes(y=after_stat('density')), color = 'steelblue')
    + facet_wrap('features')
)


# In[ ]:




