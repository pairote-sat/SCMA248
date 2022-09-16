#!/usr/bin/env python
# coding: utf-8

# ##  Assignment 1 (Exercise for Chapter 3)
# 
# This exercises is design to assist you how to use the pandas package to import, preprocess data and perform basic statistical analysis. Later we should see how data generating events can produce data of interest to insurance analysts.
# 
# We will look at the Local Government Property Insurance Fund in this chapter. The fund insures property owned by municipal governments, such as schools and libraries.
# 
# * government buildings,
# 
# * educational institutions,
# 
# * public libraries, and
# 
# * motor vehicles.
# 
# Over a thousand local government units are covered by the fund, which charges about \$25 million in annual premiums and provides insurance coverage of about \$75 billion.

# **Instructions:**
# 
# Due date: 20 January 2022 before 4 pm. 
# 
# Points: 5
# 
# 1. **Do not alter this file.**
# 2. Form a group of two members.
# 3. Duplicate this file and move it into the Personal folder (you may share access to your team member).
# 4. **Rename** the file as Assignment1_id1_id2 (id1 and id2 are student ID numbers in ascending order, e.g. Assignment1_6305001_6305001).
# 5. Write Python commands to answer each of the questions. 
# 6. For each question that **requires numerical values** (not list or dataframe), you also need to **assign the variable** e.g. ans1 to store the numerical answer for question 1. If there is more than 1 answer required, you must create more variables e.g. ans1_1, ans1_2 to store the values of the answers.
# 7. When you want to submit your file, you simply share access with me using my email pairote.sat@mahidol.edu. **Do not move your file into the DS@MathMahidol team**.

# **Exercises** Import the claim dataset namely ClaimsExperienceData.csv from my Github repository. Then write Python commands to answer the following questions.
# 
# Follow the link below for more detailed attributes and methods of pandas:
# 
# https://www.w3resource.com/python-exercises/pandas/index.php

# 0. Enter student IDs of your group.

# In[1]:


id1 = '6305001'
id2 = '6305002'


# 1. How many claims observations are there in this dataset? 

# In[ ]:





# 2. How many variables (features) are there in this dataset? List (print out) all the features. 

# In[ ]:





# 
# ## Description of Rating Variables
# 
# One of the important tasks of insurance analysts is to develop models to represent and manage the two outcome variables, **frequency** and **severity**. 
# 
# However, when actuaries and other financial analysts use those models, they do so in the context of external variables. 
# 
# In general statistical terminology, one might call these explanatory or predictor variables.
# 
# Because of our insurance focus, we call them **rating variables** as they are useful in setting insurance rates and premiums.
# 
# The following table describes the rating variables considered.
# 
# These are variables that you think might naturally be related to claims 
# outcomes.
# 
# See https://docs.google.com/spreadsheets/d/1DDFO93sGVfA3N-jYLs5yoqRXvZTFNvTmp8dguqo0qlE/edit#gid=0 for more details.
# 
# <!-- To handle the skewness, we henceforth focus on logarithmic transformations of coverage and deductibles. -->
# 
# <!-- For our immediate purposes, the coverage is our first rating variable. Other things being equal, we would expect that policyholders with larger coverage have larger claims. We will make this vague idea much more precise as we proceed, and also justify this expectation with data. -->
# 
# **Variable**  | **Description**
# ----- | -------------
# EntityType    | Categorical variable that is one of six types: (Village, City, County, Misc, School, or Town) 
# LnCoverage    | Total building and content coverage, in logarithmic millions of dollars
# LnDeduct      | Deductible, in logarithmic dollars
# AlarmCredit   | Categorical variable that is one of four types: (0, 5, 10, or 15) for automatic smoke alarms in main rooms
# NoClaimCredit | Binary variable to indicate no claims in the past two years
# Fire5         | Binary variable to indicate the fire class is below 5 (The range of fire class is 0 to 10)  

# **In what follows, for illustrate, we will consider claims data in year 2010.**

# 3. How many policies are there in 2010? 
# 
# Name the answer with the variable name **num_policies**. 
# 
# Hint: one may use `.value_counts` method that return a Series containing counts of unique values. Alternatively, you want to count False and True separately you can use `pd.Series.sum()` + `~`.

# In[2]:


# Put your Python commands here.



num_policies = 


# Do not forget to assign the variable ans3.

ans3 = 


# 4. How many claims are there in 2010? Assign the result to the variable **num_claims**.

# 5. Which policy number has the maximum number of claims and what is this claims number? Create ans5_1 to store the policy number and ans5_2 for the required claims number.

# With `.idxmax()`, we can return the index at which maximum weight value is present.
# 
# See https://www.geeksforgeeks.org/get-the-index-of-maximum-value-in-dataframe-column/.

# 6. Calculate the percentage of policyholders who did not have any claims (use the name **num_policies_no_claims** for your output).

# 7. Calculate the percentage of policyholders who had only one claim.

# 8. Calculate the average number of claims for this sample. 

# 9. The `describe()` method is used for calculating some statistical data like percentile, mean and std of the numerical values of the Series or DataFrame. 
# 
# Applying to year 2010, what do we get when we run the command claims.describe()? **Do not need to create ans9**.

# 10. A common method for determining the severity distribution is to look at the distribution of the sample of 1,377 claims. Another typical strategy is to look at the **distribution of average claims among policyholders who have made claims**.
# 
# In our 2010 sample, how many such policyholders who have made claims?

# 11. The average claim for the 209 policyholders who had only one claim is the same as the single claim they had. 
# 
# Write the command(s) to list the average claim of such 209 policyholders. **Do not need to create ans11**.

# 12. Calculate the average claim of the policyholder with the maximum number of claims.

# In[ ]:





# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=3527f36d-518e-4043-8fe3-14e8fae5803c' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
