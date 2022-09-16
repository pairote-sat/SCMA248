#!/usr/bin/env python
# coding: utf-8

# # Python Basics

# In[1]:


from IPython.display import Javascript


# In[2]:


test = Javascript('alert("hello")')
display(test)


# ## Comments
# 
# A comment is a piece of text that is not executed within a program. It can be used to provide additional information to help with code comprehension.
# 
# A comment is started with the # character and continues until the end of the line.

# In[3]:


# This is a comment line.


# ## Whitespace Formatting
# 
# Curly brackets are used to separate code blocks in many languages. Indentation is used in Python:
# This makes Python code very readable, but it also implies that formatting must be done carefully.

# In[4]:


for i in [1,2]:
    for j in [1,2]:
        print(i+j)


# Inside parentheses and brackets, whitespace is ignored, which is useful for long-winded calculations:

# In[5]:


my_list = [1, 2, 3]
my_list_of_lists = [[1,2],
                    [3,4]]


# ## Python Data Types
# 
# In Python, every value is referred to as a "object." And each object has its own data type. The following are the three most common data types:
# 
# ### Integers 
# 
# Integers are integer numbers that can be used to represent objects such as "number 8."

# In[6]:


# Example integer numbers

a = 2
b = 1
c = -1


# ### Floating-point numbers 
# 
# Floating-point numbers are a type of number that can be used to represent floating-point values.

# In[7]:


# Example integer numbers

a = 2
b = 1
c = -1


# In[8]:


# import the math module 
import math 

# Code to compute solution to the quadratic equation of the form a x^2 + b x + c = 0
sol1 = (-b + math.sqrt(b**2 - 4*a*c))/(2*a)
sol2 = (-b - math.sqrt(b**2 - 4*a*c))/(2*a)


# In[9]:


print([sol1,sol2])


# The build-in Python type() function returns the type of these objects.

# In[10]:


type(sol1)


# ### Strings 
# 
# A string is used to define a sequence of characters. Consider the word "hello." Strings are **immutable** in Python 3. You can't modify it afterwards if you've previously defined one.

# In[11]:


my_string = "Hello world"
my_string.replace("world", " Mahidol")


# In[12]:


print(my_string)


# While commands like replace() and join() can modify a string, they generate a copy of it and apply the changes to it rather than rewriting the original.

# #### How to Create a String in Python
# 
# Using single, double, or triple quotations, you can make a string in three different ways. Here's an example of each possibility:

# In[13]:


second_string = 'Mahidol University'


# The print() function can then be used to print your string in the console window. This allows you to go through your code and make sure everything works properly.
# Here's a sample for you:

# In[14]:


print(second_string)


# In[15]:


third_string = '''Department of Mathematics, 
Faculty of Science,
Mahidol University'''


# In[16]:


print(third_string)


# #### Concatenating strings
# 
# The next skill you can learn is concatenation, which is a method of joining two strings using the "+" operator. Here's how you do it:

# In[17]:


my_string + " from " +  second_string


# Note that the + operator cannot be used on two different data types, such as string and integer. You'll get the following Python error if you try it:

# #### Escaping Characters
# 
# In a Python string, backslashes (\) are used to escape characters.
# 
# For example, the given code can be used to print a string containing quote marks.

# In[18]:


# Quote from Albert Einstein 

"\"Imagination is the highest form of research\". - Albert Einstein" 


# #### String Indexing and Slicing
# 
# Because strings are lists of characters, Python strings can be indexed using the same notation as lists. Bracket notation ([index]) can be used to access a single character, or slicing can be used to access a substring ([start:end]).
# 
# Indexing with negative numbers counts from the end of the string.

# In[19]:


print(my_string[0])
print(my_string[0:5])
print(my_string[-5:])


# #### Iterate String
# 
# To iterate through a string in Python, “for…in” notation is used.

# In[20]:


for c in second_string[:8]:
    print(c)


# ## Python Variables
# 
# Variables are containers for storing data values. Variable names are case-sensitive.
# 
# Variables in Python 3 are special symbols that assign a specific storage location to a value that’s tied to it. In essence, variables are like special labels that you place on some value to know where it’s stored.
# 
# The code below demonstrates how to store a string in a variable.

# In[21]:


my_string = "Hello world"


# Let’s break it down a bit further:
# 
# * my_string is the variable name.
# * = is the assignment operator.
# * “Hello world” is a value you tie to the variable name.
# 

# Variables do not need to be declared with any particular type, and can even change type after they have been set.

# In[22]:


x = 10       # x is of type int
x = "SCMA"   # x is now of type str
print(x)


# ### Casting
# 
# Casting can be used to specify the data type of a variable.

# In[23]:


x = str(10)
type(x)


# In[24]:


y = float(10)
type(y)


# In[25]:


z = int(10.1)
z


# ### Variable Names
# 
# A variable can have a short name (such as x and y) or a longer name (such as age, carname, or total volume). Variables in Python have the following rules:
# 
# * The name of a variable must begin with a letter or the underscore character.
# 
# * A number cannot be the first character in a variable name.
# 
# * Only alpha-numeric characters and underscores (A-z, 0-9, and _) are allowed in variable names.
# * Case matters when it comes to variable names (age, Age and AGE are three different variables)

# ## Lists
# 
# In Python, lists are another important data type for specifying an ordered series of elements. In particular, they allow you to group related data and perform the same operations on multiple variables at once. Lists, unlike strings, are mutable (that is, they may be changed).
# 
# Each value in a list is referred to as an item, and it is enclosed in square brackets [ ], separated by commas. It is good practice to put a space between the comma and the next value. The values in a list do not need to be unique (the same value can be repeated).
# 
# Empty lists do not contain any values within the square brackets.

# In[26]:


my_list = [1, 2, 3]


# Alternatively, you can perform the same thing with the list() function:

# In[27]:


third_list = list((1 , 2, 3))
print(third_list)


# In[28]:


my_list == third_list


# In Python, lists are a versatile data type that can contain multiple different data types within the same square brackets. The possible data types within a list include numbers, strings, other objects, and even other lists.

# In[29]:


second_list = ["a", 2, "e", 4, "i", 6, "o", 8, "u"]


# ### How to Add Items to a List
# 
# You can add new items to existing lists in two methods. The first involves the use of the append() method:
# 
# The insert() method can be used to add an item to the specified index:

# In[30]:


my_list = [1, 2, 3]
my_list.append(4)
print(my_list)


# In[31]:


my_list.insert(2, 2.5)
print(my_list)


# ### How can I remove an item from a list?
# 
# You can do it in a variety of ways. 
# 
# * To begin, use the remove() method.
# 
# * You can also use the pop() method. If no index is supplied, the last item will be removed.
# 
# * The final way is to remove a specific item using the "del" keyword. If you want to scrap the entire list, you can use del.

# In[32]:


my_list.remove(2.5)
print(my_list)


# In[33]:


my_list.pop(1)
print(my_list)


# In[34]:


my_list = [0, 1, 2, 3, 4]
del my_list [3]
print(my_list)


# In[35]:


my_list = [0, 1, 2, 3, 4]
del my_list 


# ### Combine two lists into one
# 
# Use the + operator to combine two lists.

# In[36]:


my_list = [0, 1, 2, 3, 4] 
second_list = ["a", 2, "e", 4, "i", 6, "o", 8, "u"]
print(my_list + second_list)


# ### Change Item Value on Your List
# 
# You can easily overwrite a value of one list items:

# In[37]:


second_list = ["a", 2, "e", 4, "i", 6, "o", 8, "u"]

second_list[0] = 1
second_list[2] = 3
second_list[4] = 5
second_list[6] = 7
second_list[8] = 9
print(second_list)


# ### Loop Through the List
# 
# Using for loop you can multiply the usage of certain items, similarly to what * operator does. Here’s an example:

# In[38]:


last_list = []
for x in range(1,4): 
    last_list += ["fruit"] 
print(last_list)


# In[39]:


for x in range(1,4): 
    print(x)


# ### Copy a List
# 
# Use the built-in copy() function to replicate your data. Alternatively, you can copy a list with the list() method.

# In[40]:


old_list = ["apple", "banana", "orange"] 
new_list = old_list 
print(new_list)


# In[41]:


old_list = ["apple", "banana", "orange"] 
new_list = old_list.copy() 
print(new_list)


# In[42]:


old_list = ["apple", "banana", "orange"] 
new_list = list(old_list) 
print(new_list)

