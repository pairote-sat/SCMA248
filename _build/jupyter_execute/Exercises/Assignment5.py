#!/usr/bin/env python
# coding: utf-8

# # Assignment 4 (Exercise for Chapter 6)
# 
# Due date: 31 March 2022 before 4 pm.
# 
# Points: 10 (+ 5 extra points)
# 
# Do not alter this file.
# 1. Form a group of two members. The group member must be different from the previous Assignments 1, 2, 3 and 4.
# 
# 2. Duplicate this file and move it into the Personal folder (you may share access to your team member). Rename the file as Assignment5_id1_id2 (id1 and id2 are student ID numbers in ascending order, e.g. Assignment5_6305001_6305001).
# 
# 3. Write Python commands to answer each of the questions. For each question that requires numerical values (not list, table or dataframe), you also need to assign the variable e.g. ans1 to store the numerical answer for question 1. If there is more than 1 answer required, you must create more variables e.g. ans1_1, ans1_2 to store the values of the answers.
# 
# 4. When you want to submit your file, you simply share access with me using my email pairote.sat@mahidol.edu and my TA p.pooy.pui.i@gmail.com. Do not move your file into the DS@MathMahidol team.

# ## Problem Statement
# 
# An automobile company wants to enter the U.S. market by setting up a manufacturing plant there and producing cars locally to compete with its American and European competitors.
# 
# The company has hired an automotive consulting firm to understand the factors on which car pricing depends. Specifically, the company wants to understand the factors that affect car pricing in the U.S. market, as these can be very different from those in the Chinese market. The company wants to know:
# 
# What variables are important in predicting the price of a car.
# How well these variables describe the price of a car
# Based on various market surveys, the consulting firm has gathered a large data set on different types of cars in the American market.

# ## Regression problems
# 
# There are five basic steps when you implement linear regression:
# 
# 1. Import the packages and classes you need.
# 
# 2. Provide data to work with and possibly perform appropriate transformations.
# 
# 3. Create a regression model and fit it with the available data.
# 
# 4. Review the results of the model fit to know if the model is satisfactory.
# 
# 5. Apply the model for predictions.
# 
# These steps are more or less general to most regression approaches and implementations.

# ### Step 1: Import the packages

# In[1]:


import pandas as pd
import numpy as np

import statsmodels.api as sm

from scipy.stats import *

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import math as m
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from scipy.stats import pearsonr,spearmanr

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.base import TransformerMixin

from plotnine import *


# ### Step 2: Provide data to work with and possibly perform appropriate transformations.

# 0. Enter student IDs of your group.

# In[ ]:





# 1. Download the following dataset, CarPrice_Assignment.csv, from this Kaggle link (save this data frame as Pandas DataFrame):
# 
# https://www.kaggle.com/datasets/hellbuoy/car-price-prediction
# 
# or from my personal Github page.
# 
# https://raw.githubusercontent.com/pairote-sat/SCMA248/main/Data/CarPrice_Assignment.csv

# In[ ]:





# #### 2.1 Data Preprocessing and Feature Engineering
# 
# **What is data preprocessing?**
# 
# It is a technique used to transform raw data into more meaningful data, or data that can be understood by a machine learning model. Real-world data is often incomplete, inconsistent, and/or lacks certain behaviors or trends and is likely to contain many errors. To address this problem, data preprocessing techniques are introduced. We will talk about some of the data preprocessing techniques, namely :
# Vectorization
# Normalization
# Treatment of missing values
# 
# 
# **What is Feature engineering?**
# 
# Feature engineering is the process of selecting, manipulating, and transforming raw data into features that can be used in supervised learning. 
# 
# In order for machine learning to work well on new tasks, it may be necessary to develop and train better features. 
# 
# Feature engineering, in simple terms, is the transformation of raw data into desired features using statistical or machine learning techniques.

# #### 2.2 Missing values
# 
# When it comes to preparing your data for machine learning, missing values are one of the most typical problems. Human error, data flow interruptions, privacy concerns, and other factors can lead to missing values. Missing values, for whatever reason, affect the performance of machine learning models.

# 2. Write python code to check whether any missing values are there in the dataset.

# In[ ]:





# #### 2.3 Handling Outliers
# 
# Outlier treatment is a technique for removing outliers from a data set. This method can be used on a variety of scales to obtain a more accurate representation of data. This has an impact on the performance of the model. Depending on the model, the impact can be large or minimal. For example, linear regression is particularly susceptible to outliers. This procedure should be completed prior to model training.

# 3. Write Python code to plot the distribution of the car prices. 

# In[ ]:





# 4. Complete the following tasks:
# 
# 4.1 Identify (or list) all values that are outliers. 
# 
# 4.2 How many outliers are there?
# 
# 4.3 What can you say about the shape of this distribution?

# In[ ]:





# 5. We will use the log transformation to convert a skewed distribution to a normal or less skewed distribution. Write Python code to add a new column called 'log_price' that contains the logarithmic base 10 of the car prices.

# In[ ]:





# 6. Write Python code to plot the distribution of the values of the `log_ price` variable. 
# 
# 6.1 After log transformation, identify (or list) all values that are outliers the values of the `log_ price` variable. 

# In[ ]:





# #### 2.4 Scaling
# 
# **Feature scaling** is one of the most common and difficult problems in machine learning, but also one of the most important if you want to do it right. 
# 
# To train a predictive model, we need data with a known set of features that must be scaled up or down as appropriate. 
# 
# If we do not have comparable scales, some of the coefficients obtained by fitting the regression model could be very large or very small compared to the other coefficients. There are two common methods of rescaling:
# 
# * Min-Max scaling
# 
# * Standardisation (mean-0, sigma-1).
# 
# After scaling, continuous features become similar in terms of range. Although this step is not required for many algorithms, it is still a good idea to perform it. 
# 
# Distance-based algorithms such as k-NN and k-Means, on the other hand, require scaled continuous features as model input. 
# 
# Here we will use the standardisation scaling.
# 
# See the following link for more details:
# https://hersanyagci.medium.com/feature-scaling-with-scikit-learn-for-data-science-8c4cbcf2daff

# 7. Perform the standardisation scaling to all numeric feature variables (not the target (or response) variable, namely `price` and `log_price`.

# In[ ]:





# #### 2.5 Correlation Analysis

# The linear correlation between variable pairs is investigated using correlation analysis. This may be accomplished by combining the `corr()` and `sns.heatmap()` functions.

# 8. Write Python code to create the correlation matrix of the dataset which has been completely preprocessed.
# 
# 

# In[ ]:





# 9. Identify (list) which numeric variables have a significant positive correlation with `log_price`.

# In[ ]:





# 10. Identify (list) which numeric variables have a significant negative correlation with `log_price`.

# In[ ]:





# #### 2.6 Exploratory data analysis
# 
# It is now time to experiment with the data and make some visualizations.
# 
# In our dataset, a **pairplot** plots pairwise relationships. The pairplot function creates a grid of Axes in which each variable in the data is shared across a single row and a single column on the y-axis.
# 
# A pairs plot shows the distribution of single variables as well as the relationships between them. Pair plots are a great way to detect trends for further study, and they're simple to create in Python!

# 11. Write Python code to create pair plots of the feature and target variables (you may choose to include only features which have a significant correlation (as obtained from questions 9 and 10).

# In[ ]:





# 12. Explain the results you obtained.

# In[ ]:





# ### Step 3: Create a regression model and fit it with the available data.
# 
# Let us start with the simplest case, simple linear regression. We will use `log_price` as the dependent variable and the feature variable with highest correlation to the output `log_price` as an independent variable.

# 13. What is the variable with the highest correlation to the output `log_price`?

# In[ ]:





# #### 3.1 Split the data into training and test subset
# 
# To check the performance of a model, you should test it with new (test) data, that is with observations not used to fit (train) the model. 

# 14. Use scikit-learn’s `train_test_split()` to split your dataset into the training and test subsets. You must specifiy the arguements `random_state=id1` and `test_size= 0.3` for the `train_test_split()` function.

# In[ ]:





# 14. Use scikit-learn’s `train_test_split()` to split your dataset into the training and test subsets. You must specifiy the arguements `random_state=id1` and `test_size= 0.3` for the `train_test_split()` function.

# In[ ]:





# 15. Use either `statsmodel` or `scikit-learn` libraries to create the simple linear regression model and fit it with the **training data**.

# In[ ]:





# ### Step 4: Get results
# 
# Once you have fitted your model, you can get the results to verify that the model is working satisfactorily and interpret it.
# 
# We will obtain the following properties of the model.

# 16. Write Python to print out the coefficient of determination (R-squared).

# In[ ]:





# 17. Write Python to print out the intercept of the model.

# In[ ]:





# 18.  Write Python to print out the slope (coefficient of the chosen feature) of the model.

# In[ ]:





# ### Step 5: Predict response
# 
# Once you have a satisfactory model, you can use it for predictions with new data.

# 19. Make predictions for the test set. Add a new column of prediction values to the test set. Show the first five rows of the test set of the following columns: 
# 
# * the feature column that used for the model,
# 
# * the `log_price`, and 
# 
# * the predictions.

# In[ ]:





# 20. For model evaluation, write Python to calculate the following quantitative measurements on **the test set**
# 
# 20.1 MAE, 
# 
# 20.2 MSE, and 
# 
# 20.3 R squared. 

# In[ ]:





# 21. (5 extra points) Write Python code to visualize the predictions of the linear regression. These may include:
# 
# * Scatter plot of actual vs predicted Values
# 
# * Density plot of actual vs Predicted Values
# 
# * Residual plot (prediction Error)
# 
# Give the graphs appropriate labels and titles.

# In[ ]:




