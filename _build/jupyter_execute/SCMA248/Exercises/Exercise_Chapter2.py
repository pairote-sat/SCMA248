#!/usr/bin/env python
# coding: utf-8

# ## Exercise for Chapter 2
# 
# This Python fundamental exercise is designed to assist Python beginners in quickly learning the abilities they need. Loops, control flow, data types, operators, list, strings, input-output, and built-in functions are all core Python topics to practice.
# 
# **Please create new cells if you want to provide the Python scripts for each of the examples. Please include your name in the comments.**

# **Example 1** Print the sum of the current and previous numbers.
# 
# Write a program that iterates through the first five numbers, printing the sum of the current and previous numbers after each iteration.

# In[ ]:





# Current number: 2 Previous Number: 1 Sum: 3

# In[1]:


#Enfant01or44
f=[1,2,3,4,5]
for i,j in enumerate(f):
    if i==0:
        print('Current number: %d Previous Number: none Sum %d'%(j,j))
    else:
        print('Current number: %d Previous Number: %d Sum %d'%(j,f[i-1],j+f[i-1]))


# In[2]:


#aa
numbers = [1,2,3,4,5]

i = 0
for x in range(1,6):
    print(numbers[i] + numbers[i - 1])
    i += 1


# In[3]:


# Pairote 


# In[4]:


# Jeep


# In[5]:


# A
f=[1,2,3,4,5]
for i in range(1,len(f)):
    print(f[i]+f[i-1])


# In[6]:


#Cinnamond

num = [1,2,3,4,5]
for i in range(len(num)):
    if i == 0 :
        print(num[i]) 
    else :
        print(num[i]+num[i-1])


# **Example 2** Print characters in a string that are present at an even index number 
# 
# Write a script that takes a user-input string and displays characters that occur at even index numbers.

# In[7]:


#Enfant01or44
f = input("Enter something : ")
for i in range(int((len(f)+1)/2)):
    print(f[2*i])


# In[ ]:


#cinnamond

n = input("Input word or sentences.")
for i,j in enumerate(n) :
    if i%2 == 0 :
        print(j)


# In[ ]:





# **Example 3** Remove the first n characters from a string
# Write a program to remove characters from a string from zero to n and return a new characters.
# 
# For example,
# 
# * `remove_chars("pynative", 4)` ensures that the output is tive. The first four characters of a string must be removed in this case.
# 
# * `remove_chars("pynative", 2)` ensures that the output is native. The first two characters of a string must be removed in this case.
# 
# Note that `n` must be smaller than the string's length.

# In[ ]:


#nyn
def remove_chars(m,n):
    if n<len(m):
        return m[n:]
    else:
        return 0
print(remove_chars("Science", 4))
print(remove_chars("Science", 10))


# In[ ]:


#Cinnaomnd

def remove_chars(i,j) :
    if j == 0 :
        return i[1:]
    elif j < len(i) and j > -1 :
        return i[j:]
    else :
        return "Error"

print(remove_chars("pynative", 4))
print(remove_chars("pynative", 2))
print(remove_chars("pynative", 0))  # remove the first letter
print(remove_chars("pynative", -2))  # remove from 0 to -2 = p,v,e ==> ynati or nothing happend


# 

# **Example 4** Return the number of substrings in a string.
# 
# Write a program that calculates the number of times the substring "Angel" appears in the provided string.
# 
# Given:
# 
# new_str = new "Angel is an excellent coder. Angel is an author. Most importantly, Angel is beautiful."

# In[ ]:


#pw
new_str = "Angel is an excellent coder. Angel is an author. Most importantly, Angel is beautiful."
print(new_str.count("Angel"))


# In[ ]:


#N0mPh0ngz
x = "Angel is an excellent coder. Angel is an author. Most importantly, Angel is beautiful."
y = 'Angel'
z = x.count(y)
print(z)


# **Example 5** Print the following pattern
# 
# Write a Python code to print out the following pattern:
# 
# 1 
# 
# 2 2
# 
# 3 3 3
# 
# 4 4 4 4
# 
# 5 5 5 5 5

# In[ ]:


# PL
for i in range(1,6):
    for j in range(i) :
        print(i, end = " ")
    print()


# In[ ]:


#Enfant01or44
for i in range(1,6):
    print((str(i)+' ')*i,'\n')


# In[ ]:


#N0mPh0ngz
x = 5
for i in range(x) :
    print((str(i+1)+' ')*(i+1) + '\n' )


# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=b6a4cd23-cb61-4077-af1d-1d1fc33bd961' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
