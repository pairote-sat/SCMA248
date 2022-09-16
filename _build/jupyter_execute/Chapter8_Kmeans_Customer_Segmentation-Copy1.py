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

