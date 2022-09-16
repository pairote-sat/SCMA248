#!/usr/bin/env python
# coding: utf-8

# ## Customer segmentation
# 
# Customer segmentation is the process of categorizing your customers based on a variety of factors. These can be personal information, 
# 
# * buying habits, 
# * demographics and 
# * so on. 
# 
# The goal of customer segmentation is to gain a better understanding of each group so you can market and promote your brand more effectively.
# 
# To understand your consumer persona, you may need to use a technique to achieve your goals. Customer segmentation can be achieved in a number of ways. One is to develop a set of machine learning algorithms. In the context of customer segmentation, this article focuses on the differences between 
# 
# * the kmeans and 
# 
# * knn algorithms.

# **Customer Segmentation**: image from segmentify
# 
# ![Customer Segmentation: image from segmentify](https://www.segmentify.com/wp-content/uploads/2021/08/Top-Customer-Segmentation-Examples-every-Marketer-Needs-to-Know.png)
# 
# ![Customer Segmentation: image from segmentify](https://www.segmentify.com/wp-content/uploads/2021/08/personalisation-has-significant-positive-effects.png)

# ### Mall Customer Data
# 
# Mall Customer Segmentation Data is a dataset from Kaggle that contains the following information:
# 
# * individual unique customer IDs, 
# 
# * a categorical variable in the form of gender, and 
# 
# * three columns of age, annual income, and spending level.
# 
# These numeric variables are our main targets for identifying patterns in customer buying and spending behaviour.
# 
# The data can be downloaded from https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

# In[1]:


# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import seaborn as sns 


# In[2]:


from plotnine import *


# In[3]:


url = 'https://raw.githubusercontent.com/pairote-sat/SCMA248/main/Data/Mall_Customers.csv'

df = pd.read_csv(url)


# In[4]:


df.head()


# The dataset contains 200 observations and 5 variables. However, below is a description of each variable:
# 
# * CustomerID = Unique ID, assigned to the customer.
# 
# * Gender = Gender of the customer
# 
# * Age = Age of the customer
# 
# * Annual Income = (k$) annual income of the customer
# 
# * Spending Score = (1-100) score assigned by the mall based on customer 
# behavior and spending type

# ### Exploratory Data Analysis

# **Exercise**
# 
# 1. Perform exploratory data analysis to understand the data set before modeling it, which includes:
# 
# * Observe the data set (e.g., the size of the data set, including the number of rows and columns),
# 
# * Find any missing values,
# 
# * Categorize the values to determine which statistical and visualization methods can work with your data set,
# 
# * Find the shape of your data set, etc.
# 
# 2. Perform feature scaling standardization in the data set when preprocessing the data for the K-Means algorithm.
# 
# 3. Implement the K-Means algorithm on the annual income and spending score variables. 
# 
# * Determine the optimal number of K based on the elbow method or the silhouette method.
# 
# * Plot the cluster boundary and clusters. 
# 
# 4. Based on the optimal value of K, create a summary by averaging the age, annual income, and spending score for each cluster. Explain the main characteristics of each cluster.
# 
# 5. (Optional) Implement the K-Means algorithm on the variables annual income, expenditure score, and age. Determine the optimal number of K and visualize the results (by creating a 3D plot).

# In[5]:


df.describe()


# In[6]:


df.dtypes


# In[7]:


df.isnull().sum()


# In[8]:


(
    ggplot(df) + 
    aes('Age') +
    geom_histogram(aes(y=after_stat('density')), binwidth = 1, color = 'black') +
    geom_density(aes(y=after_stat('density')),color='blue')
)


# In[9]:


df_melted = pd.melt(df[['Age','Annual Income (k$)','Spending Score (1-100)']], var_name = 'features',value_name = 'value')


# In[10]:


df_melted


# In[11]:


(
    ggplot(df_melted) + aes('value') + geom_histogram(aes(y=after_stat('density')), color = 'lightskyblue', fill = 'lightskyblue', bins = 15)
    + geom_density(aes(y=after_stat('density')), color = 'steelblue')
    + facet_wrap('features')
)


# In[12]:


# Make a copy of the original dataset
df_model = df.copy()

# Data scaling

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df_scaled = df_model
features = [['Age','Annual Income (k$)','Spending Score (1-100)']]
for feature in features:
    df_scaled[feature] = scaler.fit_transform(df_scaled[feature])


# In[13]:


df_scaled


# ### Clustering based on two features: annual income and spending score
# 
# The figure below do appear to be some patterns in the data.

# In[14]:


#df_scaled['Gender'].tolist()


# In[15]:


#df_scaled.plot.scatter("Annual Income (k$)","Spending Score (1-100)", c = df_scaled['Gender'])

fig, ax = plt.subplots()

colors = {'Male':'lightblue', 'Female':'pink'}

ax.scatter(df_scaled['Annual Income (k$)'], df_scaled['Spending Score (1-100)'], c=df['Gender'].map(colors))

plt.title('Annual income vs spending score w.r.t gender')

plt.show()


# In[16]:


sns.scatterplot('Annual Income (k$)', 'Spending Score (1-100)', data=df_scaled, hue='Gender').set(title='Annual income vs spending score w.r.t gender')


# In[17]:


( 
    ggplot(df_scaled) 
    + aes(x = 'Annual Income (k$)', y = 'Spending Score (1-100)', color = 'Gender') 
    + geom_point()
    + labs(title = 'Annual income vs spending score w.r.t gender')
)


# #### Choosing the appropriate number of clusters: Elbow Method

# In[18]:


# Taking the annual income and spending score
features = ['Annual Income (k$)','Spending Score (1-100)']
model1 = df_scaled[features]


# fitting multiple k-means algorithms and storing the values in an empty list

SSE = []
for cluster in range(1,10):
    kmeans = KMeans(n_clusters = cluster)
    kmeans.fit(model1)
    SSE.append(kmeans.inertia_)

# converting the results into a dataframe and plotting them
frame = pd.DataFrame({'Cluster':range(1,10), 'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')


# The best number of clusters for our data is clearly 5, as the curve slope is not severe enough after that.

# #### Choosing the appropriate number of clusters: Silhouette Method
# 
# The silhouette method calculates each point's silhouette coefficients, which measure how well a data point fits into its assigned cluster based on two factors:
# 
# * How close the data point is to other points in the cluster.
# 
# * How far the data point is from points in other clusters.

# In[19]:


from yellowbrick.cluster import SilhouetteVisualizer


# In[20]:


# Instantiate the clustering model and visualizer

kmeans = KMeans(
                init="random",
                n_clusters=5,
                n_init=10,
                max_iter=300,
                random_state=88)

visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick')
# Fit the data to the visualizer
visualizer.fit(model1)  
# Compute silhouette_score
print('The silhouette score:', visualizer.silhouette_score_)
# Finalize and render the figure
visualizer.show()          

    

# For scatter plot 
kmeans.fit(model1) 
pred1 = kmeans.predict(model1)
#Getting the Centroids
centroids = kmeans.cluster_centers_
u_labels = np.unique(kmeans.predict(model1))
 
#plotting the results:
for i in u_labels:
    plt.scatter(model1.iloc[pred1 == i , 0] , model1.iloc[pred1 == i , 1] , label = i)

plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
plt.legend()
plt.show()    


# `n_clusters=5` has the best average silhouette score of around 0.55, and all clusters are above the average, indicating that it is a good choice.

# In[21]:


df['cluster'] = pred1
df.groupby('cluster').mean()[['Age','Annual Income (k$)','Spending Score (1-100)']]


# #### Cluster Analysis
# 
# Based on the above results, which clusters should be the target group?
# 
# There are five clusters created by the model including
# 
# 1. **Cluster 0: Low annual income, high spending (young age spendthrift)** 
# 
# Customers in this category earn less but spend more. People with low income but high spending scores can be viewed as possible target customers. We can see that people with low income but high spending scores are those who, for some reason, love to buy things more frequently despite their limited income. Perhaps it's because these people are happy with the mall's services. The shops/malls **may not be able to properly target these customers**, but they will not be lost.
# 
# 2.  **Cluster 1: High annual income, low spending (miser)**
# 
# Customers in this category earn a lot of money while spending little. It's amazing to observe that customers have great income yet low expenditure scores. Perhaps they are the customers who are **dissatisfied with the mall's services**. These are likely to be the mall's primary objectives, as they have the capacity to spend money. As a result, mall officials will **attempt to provide additional amenities in order to attract** these customers and suit their expectations.
# 
# 3. **Cluster 2: Low annual income, low spending (pennywise)**
# 
# They make less money and spend less money. Individuals with low yearly income and low expenditure scores are apparent, which is understandable given that people with low wages prefer to buy less; in fact, these are the smart people who know how to spend and save money. People from this cluster will be of little interest to the shops/mall.
# 
# 
# 
# 4. **Cluster 3: Medium annual income, medium spending**
# 
# In terms of income and spending, customers are average. We find that people have average income and average expenditure value. These people will **not be the primary target** of stores or malls, but they will be taken into account and other data analysis techniques can be used to increase their spending value.
# 
# 5. **Cluster 4: High annual income, high spending (young age wealthy and target customer)**
# 
# Target Customers that earn a lot of money and spend a lot of money. A **target consumer** with a high annual income and a high spending score. People with high income and high spending scores are great customers for malls and businesses because they are the primary profit generators. These individuals may be regular mall patrons who have been persuaded by the mall's amenities.

# In[ ]:





# In[ ]:





# In[ ]:





# ### Clustering based on three features: annual income, spending score and age
# 
# 

# #### Ploting the relation between age and other features

# In[22]:


( 
    ggplot(df_scaled) 
    + aes(x = 'Age', y = 'Spending Score (1-100)', color = 'Gender') 
    + geom_point()
    + labs(title = 'Age vs spending score w.r.t gender')
)


# In[23]:


( 
    ggplot(df_scaled) 
    + aes(x = 'Age', y = 'Annual Income (k$)', color = 'Gender') 
    + geom_point()
    + labs(title = 'Age vs annual income w.r.t gender')
)


# **Note** Using a 2D visualization, we can't see any distinct patterns in the data set.

# #### The elbow method

# In[24]:


# Taking the annual income and spending score
features = ['Age','Annual Income (k$)','Spending Score (1-100)']
model2 = df_scaled[features]


# fitting multiple k-means algorithms and storing the values in an empty list

SSE = []
for cluster in range(1,10):
    kmeans = KMeans(n_clusters = cluster)
    kmeans.fit(model2)
    SSE.append(kmeans.inertia_)

# converting the results into a dataframe and plotting them
frame = pd.DataFrame({'Cluster':range(1,10), 'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')


# #### The Silhouette Method

# In[25]:


from sklearn import metrics


# In[26]:


# Instantiate the clustering model and visualizer

kmeans = KMeans(
                init="random",
                n_clusters=6,
                n_init=10,
                max_iter=300,
                random_state=88)

visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick')
# Fit the data to the visualizer
visualizer.fit(model2)  
# Compute silhouette_score
print('The silhouette score:', visualizer.silhouette_score_)
# Finalize and render the figure
visualizer.show()          

    

# For scatter plot 
kmeans.fit(model2) 
pred2 = kmeans.predict(model2)
#Getting the Centroids
centroids = kmeans.cluster_centers_
u_labels = np.unique(kmeans.predict(model2))
 
#plotting the results:
for i in u_labels:
    plt.scatter(model1.iloc[pred2 == i , 0] , model1.iloc[pred2 == i , 1] , label = i)

plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
plt.legend()
plt.show()    


# In[27]:


silhouette_score = []
for cluster in range(2,10):      
    kmeans = KMeans(
                init="random",
                n_clusters=cluster,
                n_init=10,
                max_iter=300,
                random_state=88)
    
    kmeans.fit(model2)
    labels = kmeans.labels_
    silhouette_score.append(metrics.silhouette_score(model2, labels))
    
# converting the results into a dataframe and plotting them
frame = pd.DataFrame({'Cluster':range(2,10), 'Silhouette_score':silhouette_score})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['Silhouette_score'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('silhouette_score')


# In[28]:


get_ipython().system('pip install plotly')


# In[29]:


import plotly as py
import plotly.graph_objs as go


# In[30]:


df


# In[31]:


# Clustering with n_clusters = 6

#algorithm = (KMeans(n_clusters = 6 ,init='k-means++', n_init = 10 ,max_iter=300, 
#                        tol=0.0001,  random_state= 111  , algorithm='elkan') )

kmeans = KMeans(
                init="random",
                n_clusters=6,
                n_init=10,
                max_iter=300,
                random_state=88)
kmeans.fit(model2)

pred2 = kmeans.predict(model2)
#Getting the Centroids
centroids2 = kmeans.cluster_centers_

#algorithm.fit(X3)
#labels3 = algorithm.labels_
#centroids3 = algorithm.cluster_centers_


# In[32]:


df


# In[33]:


# x and y given as array_like objects
import plotly.express as px
fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
fig.show()


# In[34]:


labels3 = kmeans.predict(model2)



df['label3'] =  labels3
trace1 = go.Scatter3d(
    x= df['Age'],
    y= df['Spending Score (1-100)'],
    z= df['Annual Income (k$)'],
    mode='markers',
     marker=dict(
        color = df['label3'], 
        size= 20,
        line=dict(
            color= df['label3'],
            width= 12
        ),
        opacity=0.8
     )
)
data = [trace1]
layout = go.Layout(
#     margin=dict(
#         l=0,
#         r=0,
#         b=0,
#         t=0
#     )
    title= 'Clusters',
    scene = dict(
            xaxis = dict(title  = 'Age'),
            yaxis = dict(title  = 'Spending Score'),
            zaxis = dict(title  = 'Annual Income')
        )
)
fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig)


# In[ ]:





# In[35]:


# x and y given as array_like objects
import plotly.express as px
fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
fig.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




