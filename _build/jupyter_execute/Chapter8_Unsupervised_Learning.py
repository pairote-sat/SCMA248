#!/usr/bin/env python
# coding: utf-8

# # Unsupervised Machine Learning: K-means Clustering

# In this chapter, we will take an in-depth look at **k-means clustering** and its components. We explain why clustering is important, how to use it, and how to perform it in Python with a real dataset.

# ## What is clustering and how does it work?
# 
# **Clustering** is a collection of methods for dividing data into groups or clusters. Clusters are roughly described as collections of data objects that are more similar to each other than data objects from other clusters.

# There are numerous cases where automatic grouping of data can be very beneficial. Take the case of developing an Internet advertising campaign for a brand new product line that is about to be launched. While we could show a single generic ad to the entire population, a far better approach would be to divide the population into groups of people with common characteristics and interests, and then show individualized ads to each group. **K-means** is an algorithm that finds these groupings in large data sets where manual searching is impossible.

# ## How is clustering a problem of unsupervised learning?
# 
# Imagine a project where you need to predict whether a loan will be approved or not. We aim to predict the loan status based on the customer's gender, marital status, income, and other factors.
# Such challenges are called **supervised learning** problems when we have a target variable that we need to predict based on a set of predictors or independent variables.
# 
# There may be cases where we do not have an outcome variable that we can predict.
# 
# **Unsupervised learning** problems are those where there is no fixed target variable. 
# 
# In clustering, we do not have a target that we can predict. We examine the data and try to categorize comparable observations into different groups. Consequently, it is a challenge of unsupervised learning. There are only independent variables and no target/dependent variable.

# **k-means clustering**: image from medium
# 
# ![k-means clustering: image from medium](https://miro.medium.com/max/1400/1*IXGsBrC9FnSHGJVw9lDhQA.png)

# ## Applications of Clustering
# 
# 1. **Customer segmentation** is the process of obtaining information about a company's customer base based on their interactions with the company. In most cases, this interaction is about their purchasing habits and behaviors. Businesses can use to create targeted advertising campaigns.
# 
# ![Customer segmentation: image from connect-x](https://scontent.fbkk2-5.fna.fbcdn.net/v/t1.6435-9/119891167_183328516607191_6403713711553954542_n.jpg?stp=cp0_dst-jpg_e15_p320x320_q65&_nc_cat=110&ccb=1-5&_nc_sid=e007fa&_nc_ohc=kWGkKC7LHLkAX_PbKiO&_nc_ht=scontent.fbkk2-5.fna&oh=00_AT9wizWUDbZXgtnsrQdfC1pP862KHo1dsFDfR6OwXRMCig&oe=626EEE3F)
# 
# 2. **Recommendation engines** The recommendation system is a popular way for offering automatic personalized product, service, and information recommendations.
# 
# The recommendation engine, for example, is widely used to recommend products on Amazon, as well as to suggest songs of the same genre on YouTube.
# 
# In essence, each cluster will be given to specific preferences based on the preferences of customers in the cluster. Customers would then receive suggestions based on cluster estimates within each cluster.
# 
# ![Recommendation engines: image from woocommerce.com](https://woocommerce.com/wp-content/uploads/2013/06/Screen-Shot-2013-06-27-at-15.07.36.png)
# 
# 
# 3. **Medical application** In the medical field, clustering has been used in gene expression experiments.  The clustering results identify groupings of people who respond to medical treatments differently.

# ## The K-means Algorithm: An Overview
# 
# This section takes you step by step through the traditional form of the k-means algorithm. This section will help you decide if k-means is the best method to solve your clustering problem.
# 
# **The traditional k-means algorithm consists of only a few steps.** 
# 
# To begin, we randomly select k centroids, where k is the number of clusters you want to use. 
# 
# **Centroids** are data points that represent the center of the cluster.
# 
# The main component of the algorithm is based on a **two-step procedure** known as **expectation maximization**. 
# 
# 1. **The expectation step (E-step):** each data point is assigned to the closest centroid. 
# 
# 2. **The maximization step (M-step):** Update the centroids (mean) as being the centre of their respective observation.
# 
# We repeat these two steps until the centroid positions do not change (or until there is no further change in the clusters)
# 
# **Note** The "E-step" or "expectation step" is so called because it updates our expectation of which cluster each point belongs to. 
# 
# The "M-step" or "maximization step" is so named because it involves maximizing a fitness function that defines the location of the cluster centers - in this case, this maximization is achieved by simply averaging the data in each cluster.

# ## K-means clustering in 1 dimension: Explained
# 
# We will learn how to cluster samples that can be put on a line, and the concept can be extended more generally. Imagine that you had some data that you could plot on a line, and you knew that you had to divide it into three clusters. Perhaps they are measurements from three different tumor types or other cell types. In this case, the data yields three relatively obvious clusters, but instead of relying on our eye, we want to see if we can get a computer to identify the same three clusters, using k-means clustering. We start with raw data that we have not yet clustered.
# 
# **Step one:** choose the number of clusters you want to identify in your data - that's the K in k-means clustering. In this case, we choose K to equal three, meaning we want to identify three clusters. There is a more sophisticated way to select a value for K, but we will get into that later. 
# 
# Images below from statquest.
# 
# 
# ![K-means clustering in 1 dimension: image from statquest](https://raw.githubusercontent.com/pairote-sat/SCMA248/main/Data/k1.png)
# 

# **Step 2:** Randomly select three unique data points - these are the first clusters.
# 
# ![K-means clustering in 1 dimension: image from statquest](https://raw.githubusercontent.com/pairote-sat/SCMA248/main/Data/k2.png)
# 
# 
# **Step 3:** Measure the distance between the first point and the three initial clusters this is the distance between the first point and the three clusters.
# 
# ![K-means clustering in 1 dimension: image from statquest](https://raw.githubusercontent.com/pairote-sat/SCMA248/main/Data/k3.png)
# 
# **Step 4:** Assign the first point to the nearest cluster. In this case, the closest cluster is the blue cluster. Then we do the same for the next point. We measure the distances and then assign the point to the nearest cluster. When all the points are in clusters, we proceed to step 5.
# 
# ![K-means clustering in 1 dimension: image from statquest](https://raw.githubusercontent.com/pairote-sat/SCMA248/main/Data/k4.png)
# 
# ![K-means clustering in 1 dimension: image from statquest](https://raw.githubusercontent.com/pairote-sat/SCMA248/main/Data/k4_2.png)
# 
# ![K-means clustering in 1 dimension: image from statquest](https://raw.githubusercontent.com/pairote-sat/SCMA248/main/Data/k4_3.png)
# 
# **Step 5:** We calculate the mean value of each cluster.
# 
# ![K-means clustering in 1 dimension: image from statquest](https://raw.githubusercontent.com/pairote-sat/SCMA248/main/Data/k5.png)
# 
# Then we repeat the measurement and clustering based on the mean values. Since the clustering did not change during the last iteration, we are done.
# 
# ![K-means clustering in 1 dimension: image from statquest](https://raw.githubusercontent.com/pairote-sat/SCMA248/main/Data/k5_2.png)

# ### The quality of the cluster assignments
# 
# Before going any further, it's important to define what we mean by a good clustering solution. What do we mean when we say "best potential clustering solution"? Consider the following diagram, which depicts two potential observations assignments to the same centroid.
# 
# ![A good clustering solution: image from medium.com](https://miro.medium.com/max/1400/1*AdIvgTEaMKu9n-vNzUOPtw.png)
# 
# 
# The first assignment is clearly better than the second, but how can we quantify this using the k-means algorithm?
# 
# A good clustering solution is the one that minimizes the total sum of these (dotted red) lines, or more formally the sum of squared error  (SSE), is the one we are aiming for. 
# 
# In mathematical language, we want to identify the centroid $C$ that minimizes: given a cluster of observations $x_i \in \{x_1, x_2,..., x_m\}$.
# 
# $$ J(x) = \sum_{i = 1}^m || x_i - C||^2.$$
# 
# Since the centroid is simply the center of its respective observations, we can calculate:
# 
# $$ C = \frac{\sum_{i = 1}^m x_i}{m}.$$
# 
# This equation provides us the sum of squared error for a single centroid $C$, but we really want to minimize the sum of squared errors for all centroids $c_j \in \{c_1, c_2,..., c_k\}$ for all observations $x_i \in \{x_1, x_2,..., x_n\}$ (here $n$ not $m$). The objective function of k-means is to minimize the total sum of squared error (SST):
# 
# $$ J(x) = \sum_{j = 1}^k \sum_{i = 1}^n || x_i^{(j)} - c_j||^2,$$
# 
# where the points $x_i^{(j)}$ are assigned to the centriod $c_j$, the closest centroid .
# 
# **Note** The nondeterministic nature of the k-means algorithm is due to the random initialization stage, which means that cluster assignments will differ if the process is run repeatedly on the same dataset. Researchers frequently execute several initializations of the entire k-means algorithm and pick the cluster assignments from the one with the lowest SSE.
# 
# ![K-means clustering in 1 dimension: image from statquest](https://raw.githubusercontent.com/pairote-sat/SCMA248/main/Data/k5_3.png)
# 
# ![K-means clustering in 1 dimension: image from statquest](https://raw.githubusercontent.com/pairote-sat/SCMA248/main/Data/C2.png)
# 
# ![K-means clustering in 1 dimension: image from statquest](https://raw.githubusercontent.com/pairote-sat/SCMA248/main/Data/C3.png)
# 

# ## K-means clustering in Python: a toy example

# In[1]:


#!pip install kneed


# In[2]:


# Importing Libraries

import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import numpy as np

import seaborn as sns 


# #### A simple example
# 
# Let us get started by using scikit-learn to generate some random data points. It has a `make_blobs()` function that generates a number of gaussian clusters. We will create a two-dimensional dataset containing three distinct blobs. We set `random_state=1` to ensure that this example will always generate the same points for reproducibility.
# 
# To emphasize that this is an **unsupervised algorithm**, we will leave the labels out of the visualization. 

# In[49]:


features, true_labels = make_blobs(
    n_samples=90,
    centers=3,
    cluster_std=2,
    random_state=1
)


# In[50]:


plt.scatter(features[:, 0], features[:, 1], s=50);


# We can also put the **features** into a pandas DataFrame to make working with it easier. We plot it to see how it appears, color-coding each point according to the cluster from which it was produced.

# In[51]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd

df = pd.DataFrame(features, columns=["x1", "x2"])
df.plot.scatter("x1", "x2")


# In[52]:


#plt.scatter(features[:, 0], features[:, 1], s=50, c=true_labels);


# **Exercise**
# 
# 1. Machine learning algorithms must take all features into account. This means that all feature values must be scaled to the same scale.
# Perform feature scaling standardization in the dataset.
# 
# 2. Perform exploratory data analysis (EDA) to analyze and explore data sets and summarize key characteristics, using data visualization methods.

# In[53]:


df.head()


# In[54]:


# Make a copy of the original dataset
# df_copy = df.copy()


# In[55]:


scaler = StandardScaler()

df_scaled = df
features = [['x1','x2']]
for feature in features:
    df_scaled[feature] = scaler.fit_transform(df_scaled[feature])


# In[56]:


df_scaled.describe()


# In[57]:


df_scaled.plot.scatter("x1", "x2")


# We can see that two or three distinct clusters. This is a great situation to apply k-means clustering in, and the results will be highly valuable.

# To facilitate the analysis of tabular data, we can use Pandas in Python to convert the data into a more computer-friendly form. 
# 
# `Pandas.melt()` converts a DataFrame from wide format to long format.
# The melt() function is useful to convert a DataFrame to a format where one or more columns are identifier variables, while all other columns considered as measurement variables are subtracted from the row axis, leaving only two non-identifier columns, variable and value.

# In[58]:


df_melted = pd.melt(df_scaled[['x1','x2']], var_name = 'features',value_name = 'value')


# In[59]:


df_melted.head()


# In[60]:


from plotnine import *


# In[61]:


(
    ggplot(df_melted) + aes('value') + geom_histogram(aes(y=after_stat('density')), color = 'lightskyblue', fill = 'lightskyblue', bins = 15)
    + geom_density(aes(y=after_stat('density')), color = 'steelblue')
    + facet_wrap('features')
)


# ### Violin and swarm plots

# In[62]:


n = 0
for cols in ['x1' , 'x2']:
    n += 1 
    plt.subplot(1 , 2 , n)
    plt.subplots_adjust(hspace = 0.5 , wspace = 0.1)
    sns.violinplot(x = cols, data = df , palette = 'vlag')
    sns.swarmplot(x = cols, data = df)
    #plt.ylabel('Gender' if n == 1 else '')
    #plt.title('Boxplots & Swarmplots' if n == 2 else '')
plt.show()


# ## Clustering using K- means
# 
# The data is now ready for clustering. Before fitting the estimator to the data, you set the algorithm parameters in the KMeans estimator class in scikit-learn. The scikit-learn implementation is adaptable, with various adjustable parameters.
# 
# The following are the parameters that were utilized in this example.
# 
# * **init** The initialization method is controlled by `init`. Setting init to "random" implements the normal version of the k-means algorithm. Setting this to "k-means++" uses a more advanced technique to speed up convergence, which you'll learn about later.
# 
# * **n_clusters** For the clustering step, `n_clusters` sets k. For k-means, this is the most essential parameter.
# 
# * **n_init** The number of initializations to do is specified by `n_init`. Because two runs can converge on different cluster allocations, this is critical. The scikit-learn algorithm defaults to running 10 k-means runs and returning the results of the one with the lowest SSE.
# 
# * **max_iter** For each initialization of the k-means algorithm, `max_iter` specifies the maximum number of iterations.
# 
# First we create a new instance of the KMeans class with the following parameters. Then passing the data, we want to fit to the `fit()` method, the algorithm will actually run. You can use nested lists, numpy arrays (as long as they have the shape (nsample,nfeatures) or Pandas DataFrames.

# In[63]:


kmeans = KMeans(
                init="random",
                n_clusters=3,
                n_init=10,
                max_iter=300,
                random_state=88)

kmeans.fit(df_scaled)


# Now that we have calculated the cluster centres, we can use the `cluster_centers_` data attribute of our model to see which clusters it has chosen.

# In[64]:


centroids = kmeans.cluster_centers_
print('Final centroids:\n', centroids)


# The sum of the squared distances of data points to their closest cluster center is another data characteristic of the model, which can be obtained by using the attribute `inertia_`. This property is used by the algorithm to determine whether it has converged. In general, a lower number indicates a better match.

# In[65]:


print('The sum of the squared distrance:\n', kmeans.inertia_)


# The cluster assignments are stored as a one-dimensional NumPy array in kmeans.labels_ or using the function `predict`. 

# In[66]:


pred = kmeans.predict(df_scaled)
print('predicted labels:\n',pred[0:5])


# In[67]:


#labels = kmeans.labels_


# In[68]:


print('predicted labels:\n',kmeans.labels_[0:5])


# In[69]:


df_scaled['cluster'] = pred
print('the number of samples in each cluster:\n', df_scaled['cluster'].value_counts())


# In[70]:


df_scaled.drop('cluster',axis='columns', inplace=True)


# ## Choosing the appropriate number of clusters: Elbow Method
# 
# The **elbow method** is a widely used approach to determine the proper number of clusters.
# 
# To perform the elbow method, we run many k-means by increasing k at each iteration and plot the SST (also known as **intetia**) as a function of the number of clusters. We will observe that it continues to decrease as we increase k. The distance between each point and its nearest centroid will decrease as more centroids are added.
# 
# The **elbow point** is a position on the SSE curve when **it begins to bend**. This point is believed to be a good compromise between error and cluster count. From the figure below, the elbow is positioned at k=3 in our example.

# In[71]:


# fitting multiple k-means algorithms and storing the values in an empty list
SSE = []
for cluster in range(1,20):
    kmeans = KMeans(n_clusters = cluster)
    kmeans.fit(df_scaled)
    SSE.append(kmeans.inertia_)

# converting the results into a dataframe and plotting them
frame = pd.DataFrame({'Cluster':range(1,20), 'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')


# From the figure, the elbow is positioned at k=3 in our example where the curve starts to bend.

# ## Ploting the cluster boundary and clusters
# 
# To plot the cluster boundary and the clusters, we run the Code below, which is modified from https://www.kaggle.com/code/irfanasrullah/customer-segmentation-analysis/notebook

# In[72]:


# Using n_clusters = 3 as suggested by the elbow method

kmeans = KMeans(
                init="random",
                n_clusters=3,
                n_init=10,
                max_iter=300,
                random_state=88)

kmeans.fit(df_scaled)


# In[73]:


X1 = df_scaled[['x1','x2']].values

h = 0.02
x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
##Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()]) 

meshed_points = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns = ['x1','x2'])
Z = kmeans.predict(meshed_points)


# In[74]:


plt.figure(1 , figsize = (15 , 7) )
plt.clf()
Z = Z.reshape(xx.shape)
plt.imshow(Z , interpolation='nearest', 
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower')

plt.scatter( x = 'x1' ,y = 'x2' , data = df_scaled , c = pred , s = 100 )
plt.scatter(x = centroids[: , 0] , y =  centroids[: , 1] , s = 200 , c = 'red' , alpha = 0.5)
plt.ylabel('x1') , plt.xlabel('x2')
plt.show()


# In[75]:


#np.c_[xx.ravel(), yy.ravel()]


# In[76]:


#np.c_[xx.ravel(), yy.ravel()][:,0]


# In[77]:


#pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns = ['x1','x2'])


# Alternatively, one can run the following Python code: (modified from https://www.askpython.com/python/examples/plot-k-means-clusters-python)

# In[78]:


#Getting the Centroids
centroids = kmeans.cluster_centers_
u_labels = np.unique(kmeans.predict(df_scaled.iloc[:,0:2]))
 
#plotting the results:
 
for i in u_labels:
    plt.scatter(df_scaled.iloc[pred == i , 0] , df_scaled.iloc[pred == i , 1] , label = i)

plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
plt.legend()
plt.show()


# In[79]:


#u_labels


# In[80]:


#df_scaled.iloc[pred == 0 , 0]
#pred == 0


# ## Choosing the appropriate number of clusters: Silhouette Method
# 
# In K-Means clustering, the number of clusters (k) is the most crucial hyperparameter. If we already know how many clusters we want to group the data into, tuning the value of k is unnecessary. 
# 
# The **silhouette method** is also used to determine the optimal number of clusters, as well as to interpret and validate consistency within data clusters. 
# 
# The silhouette method calculates each point's silhouette coefficients, which measure how well a data point fits into its assigned cluster based on two factors:
# 
# * How close the data point is to other points in the cluster.
# 
# * How far the data point is from points in other clusters.
# 
# This method also displays a clear graphical depiction of how effectively each object has been classified
# 
# The silhouette value is a  measure of how similar an object is to its own cluster (cohesion) compared to other clusters (separation). 

# ### Steps to find the silhouette score
# 
# To find the silhouette coefficient of each point, follow these steps:
# 
# 1. For data point  ${\displaystyle i}$ in the cluster ${\displaystyle C_{I}}$, we calculate the mean distance between ${\displaystyle i}$ and all other data points in the same cluster 
# 
# $${\displaystyle a(i)={\frac {1}{|C_{I}|-1}}\sum _{j\in C_{I},i\neq j}d(i,j)},$$
# 
# This ${\displaystyle a(i)}$ measures how well ${\displaystyle i}$ is assigned to its cluster (the smaller the value, the better the assignment).
# 
# 2. We then calculate the mean dissimilarity of point ${\displaystyle i}$ to its **neighboring cluster**.
# 
# $${\displaystyle b(i)=\min _{J\neq I}{\frac {1}{|C_{J}|}}\sum _{j\in C_{J}}d(i,j)}$$
# 
# 3. We now define a silhouette coefficient of one data point ${\displaystyle i}$
# 
# $${\displaystyle s(i)={\frac {b(i)-a(i)}{\max\{a(i),b(i)\}}}}$$
# 
# 4. To determine the **silhouette score**, compute silhouette coefficients for each point and average them over all samples.
# 
# The silhouette value varies from [1, -1], with a high value indicating that the object is well matched to its own cluster but poorly matched to nearby clusters. 
# 
# * The clustering setup is useful if the majority of the objects have a high value. 
# 
# * The clustering setup may have too many or too few clusters if many points have a low (the sample is very close to the neighboring clusters) or negative value (the sample is assigned to the wrong clusters).

# ![silhouette coefficients: image from medium.com](https://miro.medium.com/max/902/1*cNzzMupO355ohnVqXnvxEA.png)

# #### Silhouette Visualizer 
# 
# The **Silhouette Visualizer** visualizes which clusters are dense and which are not by displaying the silhouette coefficient for each sample on a per-cluster basis. This is especially useful for determining cluster imbalance or comparing different visualizers to determine a K value.
# 
# We will import the **Yellowbrik** library for silhoutte analsis, an extension to the Scikit-Learn API that simplifies model selection and hyperparameter tuning.
# 
# For more details about the library, please refer to
# https://www.scikit-yb.org/en/latest/.
# 
# The following results compares different visualizers with n_cluster = 2, 3 and 4.

# In[81]:


from yellowbrick.cluster import SilhouetteVisualizer


# In[82]:


# Instantiate the clustering model and visualizer

kmeans = KMeans(
                init="random",
                n_clusters=2,
                n_init=10,
                max_iter=300,
                random_state=88)

visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick')
# Fit the data to the visualizer
visualizer.fit(df_scaled)  
# Compute silhouette_score
print('The silhouette score:', visualizer.silhouette_score_)
# Finalize and render the figure
visualizer.show()          

        

# For scatter plot 
kmeans.fit(df_scaled) 
pred2 = kmeans.predict(df_scaled)
#Getting the Centroids
centroids = kmeans.cluster_centers_
u_labels = np.unique(kmeans.predict(df_scaled.iloc[:,0:2]))
 
#plotting the results:
for i in u_labels:
    plt.scatter(df_scaled.iloc[pred2 == i , 0] , df_scaled.iloc[pred2 == i , 1] , label = i)

plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
plt.legend()
plt.show()


# In[83]:


# Instantiate the clustering model and visualizer

kmeans = KMeans(
                init="random",
                n_clusters=3,
                n_init=10,
                max_iter=300,
                random_state=88)

visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick')
# Fit the data to the visualizer
visualizer.fit(df_scaled)  
# Compute silhouette_score
print('The silhouette score:', visualizer.silhouette_score_)
# Finalize and render the figure
visualizer.show()          

         

# For scatter plot 
kmeans.fit(df_scaled) 
pred3 = kmeans.predict(df_scaled)
#Getting the Centroids
centroids = kmeans.cluster_centers_
u_labels = np.unique(kmeans.predict(df_scaled.iloc[:,0:2]))
 
#plotting the results:
for i in u_labels:
    plt.scatter(df_scaled.iloc[pred3 == i , 0] , df_scaled.iloc[pred3 == i , 1] , label = i)

plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
plt.legend()
plt.show()


# In[84]:


# Instantiate the clustering model and visualizer

kmeans = KMeans(
                init="random",
                n_clusters=4,
                n_init=10,
                max_iter=300,
                random_state=88)


visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick')
# Fit the data to the visualizer
visualizer.fit(df_scaled)  
# Compute silhouette_score
print('The silhouette score:', visualizer.silhouette_score_)
# Finalize and render the figure
visualizer.show()          



# For scatter plot 
kmeans.fit(df_scaled) 
pred4 = kmeans.predict(df_scaled)

#Getting the Centroids
centroids = kmeans.cluster_centers_
u_labels = np.unique(kmeans.predict(df_scaled.iloc[:,0:2]))
 
#plotting the results:
for i in u_labels:
    plt.scatter(df_scaled.iloc[pred4 == i , 0] , df_scaled.iloc[pred4 == i , 1] , label = i)

plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
plt.legend()
plt.show()


# Based on the above results, we observe the following:
# 
# * At **n_clusters=4**, the silhouette plot shows that the n_cluster value of 4 is a poor choice because most points in the cluster with cluster_label=2 have below average silhouette values.
# 
# * At **n_clusters=2**, the thickness of the silhouette graph for the cluster with cluster_label=1 is larger due to the grouping of the 2 sub-clusters into one large cluster. Moreover, n_clusters=2 has the best average silhouette score of around 0.65, and all clusters are above the average, indicating that it is a good choice.
# 
# * For **n_clusters=3**, all plots are more or less similar in thickness and therefore similar in size, which can be considered the one of the optimal values.
# 
# * Silhouette analysis is more inconclusive in deciding between 2 and 3.
# 
# The bottom line is that **good n clusters** will have a silhouette average score far above 0.5, and all of the clusters will have a score higher than the average.

# In[ ]:





# ### Evaluating Clustering Performance Using Advanced Techniques: Adjusted Rand Index
# 
# 
# Without the use of ground truth labels, the elbow technique and silhouette coefficient evaluate clustering performance. 
# 
# **Ground truth labels** divide data points into categories based on a human's or an algorithm's classification. When used without context, these measures do their best to imply the correct number of clusters, but they can be misleading.
# 
# Note that datasets with ground truth labels are uncommon in practice.

# ##### Adjusted Rand Index 
# 
# Because the ground truth labels are known, a clustering metric that considers labels in its evaluation can be used. The scikit-learn version of a standard metric known as the **adjusted rand index (ARI)** can be used. 
# 
# The Rand Index computes a similarity measure between two clusterings by evaluating all pairs of samples and counting pairs that are assigned in the same or different clusters in the predicted and true clusterings,
# 
# The Rand index has been adjusted to account for chance. For more information, please refer to https://towardsdatascience.com/performance-metrics-in-machine-learning-part-3-clustering-d69550662dc6.
# 
# The output values of the ARI vary from -1 to 1. A score around 0 denotes random assignments, whereas a score near 1 denotes precisely identified clusters.

# In[85]:


from sklearn.metrics import adjusted_rand_score


# Compare the clustering results with n_clusters= 2 and 3 using ARI as the performance metric.

# In[86]:


print('ARI with n_clusters=2:',adjusted_rand_score(true_labels, pred2))
print('ARI with n_clusters=3:',adjusted_rand_score(true_labels, pred3))


# The silhouette coefficient was misleading, as evidenced by the above output. In comparison to k-means, using 3 clusters is the better solution for the dataset.
# 
# The quality of clustering algorithms can also be assessed using a variety of measures. See the following link for more details: https://scikit-learn.org/stable/modules/model_evaluation.html

# **Final Remarks: Challenges with the K-Means Clustering Algorithm**
# 
# One of the most common challenges when working with K-Means is that the size of the clusters varies. See https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/.
# 
# Another challenge with K-Means is that the densities of the original points are different. Let us assume these are the original points:
# 
# One of the solutions is to use a higher number of clusters. So, in all the above scenarios, instead of using 3 clusters, we can use a larger number. Perhaps setting k=10 will lead to more meaningful clusters.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[87]:


centroids


# In[88]:


pred = kmeans.predict(df_scaled)
df_scaled['cluster'] = pred
print('the number of samples in each cluster:\n', df_scaled['cluster'].value_counts())


# In[89]:


df_scaled.drop('cluster',axis='columns', inplace=True)


# In[90]:


visualizer.silhouette_score_


# In[91]:


#silhouette_score(X, cluster_labels)
silhouette_score(df_scaled, kmeans.labels_)


# In[92]:


# fitting multiple k-means algorithms and storing the values in an empty list
silhouette_scores = []
for cluster in range(2,20):
    kmeans = KMeans(n_clusters = cluster)
    kmeans.fit(df_scaled)
    silhouette_scores.append(silhouette_score(df_scaled, kmeans.labels_)

)

# converting the results into a dataframe and plotting them
frame = pd.DataFrame({'Cluster':range(2,20), 'Silhouette_scores':silhouette_scores})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['Silhouette_scores'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette_scores')


# In[ ]:





# In[ ]:





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

# In[93]:


# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import seaborn as sns 


# In[94]:


from plotnine import *


# In[95]:


url = 'https://raw.githubusercontent.com/pairote-sat/SCMA248/main/Data/Mall_Customers.csv'

df = pd.read_csv(url)


# In[96]:


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

# In[97]:


df.describe()


# In[98]:


df.dtypes


# In[99]:


df.isnull().sum()


# In[100]:


(
    ggplot(df) + 
    aes('Age') +
    geom_histogram(aes(y=after_stat('density')), binwidth = 1, color = 'black') +
    geom_density(aes(y=after_stat('density')),color='blue')
)


# In[101]:


df_melted = pd.melt(df[['Age','Annual Income (k$)','Spending Score (1-100)']], var_name = 'features',value_name = 'value')


# In[102]:


df_melted


# In[103]:


(
    ggplot(df_melted) + aes('value') + geom_histogram(aes(y=after_stat('density')), color = 'lightskyblue', fill = 'lightskyblue', bins = 15)
    + geom_density(aes(y=after_stat('density')), color = 'steelblue')
    + facet_wrap('features')
)


# In[104]:


# Make a copy of the original dataset
df_model = df.copy()

# Data scaling

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df_scaled = df_model
features = [['Age','Annual Income (k$)','Spending Score (1-100)']]
for feature in features:
    df_scaled[feature] = scaler.fit_transform(df_scaled[feature])


# In[105]:


df_scaled


# ### Clustering based on two features: annual income and spending score
# 
# The figure below do appear to be some patterns in the data.

# In[106]:


#df_scaled['Gender'].tolist()


# In[107]:


#df_scaled.plot.scatter("Annual Income (k$)","Spending Score (1-100)", c = df_scaled['Gender'])

fig, ax = plt.subplots()

colors = {'Male':'lightblue', 'Female':'pink'}

ax.scatter(df_scaled['Annual Income (k$)'], df_scaled['Spending Score (1-100)'], c=df['Gender'].map(colors))

plt.title('Annual income vs spending score w.r.t gender')

plt.show()


# In[108]:


sns.scatterplot('Annual Income (k$)', 'Spending Score (1-100)', data=df_scaled, hue='Gender').set(title='Annual income vs spending score w.r.t gender')


# In[109]:


( 
    ggplot(df_scaled) 
    + aes(x = 'Annual Income (k$)', y = 'Spending Score (1-100)', color = 'Gender') 
    + geom_point()
    + labs(title = 'Annual income vs spending score w.r.t gender')
)


# #### Choosing the appropriate number of clusters: Elbow Method

# In[110]:


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

# In[111]:


from yellowbrick.cluster import SilhouetteVisualizer


# In[112]:


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

# In[113]:


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

# In[114]:


( 
    ggplot(df_scaled) 
    + aes(x = 'Age', y = 'Spending Score (1-100)', color = 'Gender') 
    + geom_point()
    + labs(title = 'Age vs spending score w.r.t gender')
)


# In[115]:


( 
    ggplot(df_scaled) 
    + aes(x = 'Age', y = 'Annual Income (k$)', color = 'Gender') 
    + geom_point()
    + labs(title = 'Age vs annual income w.r.t gender')
)


# **Note** Using a 2D visualization, we can't see any distinct patterns in the data set.

# #### The elbow method

# In[116]:


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

# In[117]:


from sklearn import metrics


# In[118]:


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


# In[119]:


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


# In[120]:


get_ipython().system('pip install plotly')


# In[121]:


import plotly as py
import plotly.graph_objs as go


# In[122]:


df


# In[123]:


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


# In[124]:


df


# In[125]:


# x and y given as array_like objects
import plotly.express as px
fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
fig.show()


# In[126]:


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





# In[127]:


# x and y given as array_like objects
import plotly.express as px
fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
fig.show()


# In[ ]:




