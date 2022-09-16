#!/usr/bin/env python
# coding: utf-8

# ## Assignment 4 (Exercise for Chapters 5)
# 
# Due date: 15 March 2022 before 4 pm.
# 
# Points: 5 + (**4 extra points**)
# 
# Do not alter this file.
# 1. Form a group of two members. The group member must be different from the previous Assignments 1, 2 and 3.
# 
# 2. Duplicate this file and move it into the Personal folder (you may share access to your team member). Rename the file as Assignment4_id1_id2 (id1 and id2 are student ID numbers in ascending order, e.g. Assignment4_6305001_6305001).
# 
# 3. Write Python commands to answer each of the questions. For each question that requires numerical values (not list, table or dataframe), you also need to assign the variable e.g. ans1 to store the numerical answer for question 1. If there is more than 1 answer required, you must create more variables e.g. ans1_1, ans1_2 to store the values of the answers.
# 
# 4. When you want to submit your file, you simply share access with me using my email pairote.sat@mahidol.edu and my TA p.pooy.pui.i@gmail.com. Do not move your file into the DS@MathMahidol team.

# **Goals:** The purpose of this exercise is to perform simple descriptive statistics and to describe the **importance of data visualisation** before analyzing and model building.

# 0. Enter student IDs of your group.

# 1. Download the following dataset from this Github link (store this data frame as a pandas DataFrame with the variable name `anscombes`): 
# 
# https://gist.github.com/ericbusboom/b2ac1d366c005cd2ed8c
# 
# The dataset consists of values $x$ and $y$ from four different groups (I,II,III and IV which are given by the column namely `dataset`).

# Perform statistical data analysis of the `anscombes` dataset by answering the following questions:
# 
# 2. Write python code to create a table that provides a summary of the variables $x$ and $y$ for each of groups (including the mean of $x$, the sample variance of $x$, the mean of $y$, the sample variance of $y$).

# 2.1 Find the mean of $x$ in the group I.

# 2.2 Find the mean of $x$ in the group II.

# 2.3 Find the sample variance of $y$ in the group III.

# 2.4 Find the sample variance of $y$ in the group IV.

# 3. Write python code to create a grouped boxplot of the values $x$ categorized by groups (I, II, III and IV). 
# 
# Note that a grouped boxplot is a boxplot where categories are organized in groups and subgroups.
# 
# See for detail: https://www.r-graph-gallery.com/265-grouped-boxplot-with-ggplot2.html#:~:text=A%20grouped%20boxplot%20is%20a,called%20in%20the%20fill%20argument.

# 4. Write python code to create a grouped boxplot of the values $y$ categorized by groups (I, II, III and IV). 

# 5. Write python code to plot (in the same figure) histograms and the kernel density estimate (KDE) of the values $x$ for each of the four datasets. (you may use `facet_wrap` in **plotnine**  to build a plot for multiple subsets of the data set) 

# 6. Write python code to plot (in the same figure) histograms and the kernel density estimate (KDE) of the values $y$ for each of the groups. (you may use `facet_wrap` in **plotnine**  to build a plot for multiple subsets)

# 7. Write python code that create a table that summarizes the (linear) correlation coefficient of the variables $x$ and $y$ for each of the groups.

# 7.1 Find the correlation coefficient of the variables $x$ and $y$ of the dataset I.

# 7.2 Find the correlation coefficient of the variables $x$ and $y$ of the dataset II.

# 8. Write python code to graphically display the relationship of the variables $x$ and $y$ for each of groups.

# 9. Discuss and conclude the results from the data analysis of this work.

# Extra questions for extra points (4 points):
# 
# Complete questions 10 to earn extra points.

# 10. Perform regression analysis to determine the possible relationship between the variables $x$ and $y$ for each group.
# 
# 10.1 Find the regression line for the data in group I (for e.g. the regression line is given as 'y = a + b x' where a and b represent the y intercept and slope of the regression line).
# 
# 10.2 Find the regression line for the data in group II.
# 
# 10.3 Find the coefficient of determination (R-squared) of the dataset I.
# 
# 10.4 Find the coefficient of determination (R-squared) of the dataset II.

# 0. Enter student IDs of your group.

# 1. Download the following dataset from this Github link (store this data frame as a pandas DataFrame with the variable name `anscombes`): 
# 
# https://gist.github.com/ericbusboom/b2ac1d366c005cd2ed8c
# 
# The dataset consists of values $x$ and $y$ from four different groups (I,II,III and IV which are given by the column namely `dataset`).

# In[1]:


import numpy as np
import pandas as pd
from plotnine import *

# Add this line so you can plot your charts into your Jupyter Notebook.
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


# We want to specify that 'id' is the index column while loading this data set with the following command (with the parameter index_col):
url = 'https://raw.githubusercontent.com/pairote-sat/SCMA248/main/Data/anscombes.csv'
#anscombes = pd.read_csv('/Users/Kaemyuijang/SCMA248/Data/anscombes.csv',index_col = 0)

anscombes = pd.read_csv(url,index_col = 0)


# In[5]:


#anscombes.dtypes
anscombes.head(5)


# Perform statistical data analysis of the `anscombes` dataset by answering the following questions:
# 
# 2. Write python code to create a table that provides a summary of the variables $x$ and $y$ for each of groups (including the mean of $x$, the sample variance of $x$, the mean of $y$, the sample variance of $y$).

# In[7]:


anscombes.groupby('dataset').describe()


# 2.1 Find the mean of $x$ in the group I.

# In[10]:


output = anscombes.groupby('dataset').describe()

output.loc['I',('x', 'mean')]

ans2_1 = output.loc['I',('x', 'mean')]

print('Q2.1 the mean of x in the group I:', ans2_1)


# 2.2 Find the mean of $x$ in the group II.

# In[11]:


ans2_2 = output.loc['II',('x', 'mean')]

print('Q2.2 the mean of x in the group II:', ans2_2)


# 2.3 Find the sample variance of $y$ in the group III.

# In[12]:


ans2_3 = output.loc['III',('y', 'std')] ** 2

print('Q2.3 the sample variance of $y$ in the group III:', ans2_3)


# 2.4 Find the sample variance of $y$ in the group IV.

# In[13]:


ans2_4 = output.loc['IV',('y', 'std')] ** 2

print('Q2.4 the sample variance of $y$ in the group IV:', ans2_4)


# 3. Write python code to create a grouped boxplot of the values $x$ categorized by groups (I, II, III and IV). 
# 
# Note that a grouped boxplot is a boxplot where categories are organized in groups and subgroups.
# 
# See for detail: https://www.r-graph-gallery.com/265-grouped-boxplot-with-ggplot2.html#:~:text=A%20grouped%20boxplot%20is%20a,called%20in%20the%20fill%20argument.

# In[14]:


( ggplot(anscombes)  + 
     aes(x = 'dataset', y = 'x') +
     geom_boxplot()  
) 


# 4. Write python code to create a grouped boxplot of the values $y$ categorized by groups (I, II, III and IV). 

# In[15]:


( ggplot(anscombes)  + 
     aes(x = 'dataset', y = 'y') +
     geom_boxplot()  
) 


# 5. Write python code to plot (in the same figure) histograms and the kernel density estimate (KDE) of the values $x$ for each of the four datasets. (you may use `facet_wrap` in **plotnine**  to build a plot for multiple subsets of the data set) 

# In[16]:


( ggplot(anscombes)  + 
     aes(x = 'x') + 
     geom_histogram(aes(y=after_stat('count'))) + 
     geom_density(aes(y=after_stat('count'))) +
     facet_wrap('dataset') 
) 


# 6. Write python code to plot (in the same figure) histograms and the kernel density estimate (KDE) of the values $y$ for each of the groups. (you may use `facet_wrap` in **plotnine**  to build a plot for multiple subsets)

# In[17]:


( ggplot(anscombes)  + 
     aes(x = 'y') + 
     geom_histogram(aes(y=after_stat('count'))) + 
     geom_density(aes(y=after_stat('count'))) +
     facet_wrap('dataset') 
) 


# 7. Write python code that create a table that summarizes the (linear) correlation coefficient of the variables $x$ and $y$ for each of the groups.

# In[18]:


anscombes.groupby('dataset').corr()


# 7.1 Find the correlation coefficient of the variables $x$ and $y$ of the dataset I.

# In[19]:


ans7_1  = anscombes.groupby('dataset').corr().loc['I'].iloc[0,1]
      
print('Q7.1 the correlation coefficient of the variables $x$ and $y$ of the dataset I:\n', ans7_1)      


# 7.2 Find the correlation coefficient of the variables $x$ and $y$ of the dataset II.

# In[20]:


ans7_2  = anscombes.groupby('dataset').corr().loc['II'].iloc[0,1]
      
print('Q7.2 the correlation coefficient of the variables $x$ and $y$ of the dataset I:\n', ans7_2)      


# 8. Write python code to graphically display the relationship of the variables $x$ and $y$ for each of groups.

# In[21]:


anscombes.head()


# In[22]:


(
    ggplot(anscombes) + 
    aes(x = 'x', y='y') + 
    geom_point() +
    facet_wrap('dataset')
)


# 9. Discuss and conclude the results from the data analysis of this work.

# Q9: **Anscombe's Quartet** is a collection of four data sets that are essentially equal in terms of simple descriptive statistics (as shown in Q2), but have some oddities in the dataset: 
# 
# These oddities trick the regression model if formed. When plotted on scatter plots (Q8), they have extremely different distributions and appear differently.

# Extra questions for extra points (4 points):
# 
# Complete questions 10 to earn extra points.

# 10. Perform regression analysis to determine the possible relationship between the variables $x$ and $y$ for each group.

# Q10: The results from the regression analysis are identical, but when graphed, they differ significantly. This dataset demonstrates the importance of plotting graphs before analysing and creating models, as well as the impact of other observations on statistical features.

# 10.1 Find the regression line for the data in group I (for e.g. the regression line is given as 'y = a + b x' where a and b represent the y intercept and slope of the regression line).

# In[69]:


import statsmodels.formula.api as smf


# In[36]:


#anscombes.query("dataset == 'I'")


# In[70]:


group1 = smf.ols(formula='y ~ x', data=anscombes.query("dataset == 'I'")).fit()
print('Q10.1: the slope and intercept of the regression line\n', group1.params)


# 10.2 Find the regression line for the data in group II.

# In[71]:


group2 = smf.ols(formula='y ~ x', data=anscombes.query("dataset == 'II'")).fit()
print('Q10.2: the slope and intercept of the regression line\n', group2.params)


# 10.3 Find the coefficient of determination (R-squared) of the dataset I.

# In[74]:


group1.rsquared


# 10.4 Find the coefficient of determination (R-squared) of the dataset II.

# In[76]:


group2.rsquared


# The regression lines for each group:

# In[78]:


(
    ggplot(anscombes) + 
    aes(x = 'x', y='y') + 
    geom_point() +
    geom_smooth(method='lm',color='green') +
    facet_wrap('dataset')
)


# In[ ]:




