#!/usr/bin/env python
# coding: utf-8

# ## Exercise for Chapter 4
# 
# According to the research result **Socioeconomic development and life expectancy relationship: evidence from the EU accession candidate countries** by Goran Miladinov in Journal of Population Sciences, Genus volume 76, Article number: 2 (2020), the results show that 
# 
# * a country's population health and socioeconomic development have a significant impact on life expectancy at birth; 
# 
# * in other words, as a country's population health and socioeconomic development improves, infant mortality rates decrease, and life expectancy at birth appears to rise. 
# 
# * Through increased economic growth and development in a country, **GDP per capita raises life expectancy at birth, resulting in a longer lifespan**.
# 
# https://genus.springeropen.com/articles/10.1186/s41118-019-0071-0#:~:text=GDP%20per%20capita%20increases%20the,to%20the%20prolongation%20of%20longevity.)

# In this section, we use data to attempt to gain insight on the relationship between life expectancy and gdp per capita of world countries.
# 
# Note that we will download a dataset from **Kaggle**. 
# 
# Kaggle, a Google LLC subsidiary, is an online community of data scientists and machine learning experts. Users can use Kaggle to search and publish data sets, study and construct models in a web-based data-science environment, collaborate with other data scientists and machine learning experts, and compete in data science competitions.

# 1. Import the Gapminder World dataset from the following link:
# 
# https://www.kaggle.com/tklimonova/gapminder-datacamp-2007?select=gapminder_full.csv
# 
# (for more detail on how to read CSV file from kaggle https://www.analyticsvidhya.com/blog/2021/04/how-to-download-kaggle-datasets-using-jupyter-notebook/)

# In[1]:


get_ipython().system('pip install opendatasets')


# In[2]:


import opendatasets as od


# In[3]:


# od.download("https://www.kaggle.com/rashikrahmanpritom/heart-attack-analysis-prediction-dataset")

od.download('https://www.kaggle.com/tklimonova/gapminder-datacamp-2007')


# In[2]:


import pandas as pd


# In[3]:


gapminder = pd.read_csv('/Users/Kaemyuijang/SCMA248/Data/gapminder_full.csv')


# In[4]:


gapminder


# We will begin by looking at some of its features to get get an idea of its content.

# 2. How many qualitative variables are there in this Gapminder dataset?
# (See for more detail:
# 
# https://www.abs.gov.au/websitedbs/D3310114.nsf/Home/Statistical+Language+-+quantitative+and+qualitative+data#:~:text=What%20are%20quantitative%20and%20qualitative,much%3B%20or%20how%20often).&text=Qualitative%20data%20are%20data%20about%20categorical%20variables%20(e.g.%20what%20type).
# 
# **Note:** It is crucial to figure out whether the data is quantitative or qualitative, as this has an impact on the statistics that can be obtained.

# 3. Write Python code to create a table that gives the number of countries in each continent of **the lastest year** in this dataset.

# 4. Write Python code to graphically present the results obtained in the previous question.

# 5. **What Is GDP Per Capita?**
# 
# The per capita gross domestic product (GDP) is a financial measure that calculates a country's economic output per person by dividing its GDP by its population.
# 
# Write Python code to summarize some statistical data like percentile, mean and standard deviation of the GDP per capita of the latest year broken down by continent.

# 6. What is the average GDP per capita in Asian countries obtained above?

# 7. Plot the histogram for per capita GDP in each continent.

# **Extra questions for extra points (4 points):**
# 
# Complete questions 8 and 9 to earn extra points.
# 
# For the following questions, you will need to check out the lecture note in Chapter 4 and you will require to preprocess the orginal data by adding regional classfication into Gapminder dataset. Then the column `group` can then be added. 
# 
# 
# 8. Append the column called `group` that groups countries into 5 different groups as follows:
# 
# * West: ["Western Europe", "Northern Europe","Southern Europe", "Northern America",
# "Australia and New Zealand"]
# 
# * East Asia: ["Eastern Asia", "South-Eastern Asia"]
# 
# * Latin America: ["Caribbean", "Central America",
# "South America"]
# 
# * Sub-Saharan: [continent == "Africa"] &
# [region != "Northern Africa"]
# 
# * Other: All remaining countries (also including NAN). 

# 9. We now want to compare the distribution across these five groups to confirm the “west versus the rest” dichotomy. To do this, we will work with the 1967 data. We could generate five histograms or five smooth density plots, but it may be more practical to have all the visual summaries **in one plot**. Write Python code to stack smooth density plots (or histograms) vertically (with slightly overlapping lines) that share the same x-axis.

# In[ ]:




