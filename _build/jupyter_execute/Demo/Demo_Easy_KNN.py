#!/usr/bin/env python
# coding: utf-8

# In[1]:


# pip list


# In[2]:


#!pip install sklearn


# In[3]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# In[4]:


url = 'https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/fruit_data_with_colors.txt'
    
fruits = pd.read_table(url)  


# In[6]:


fruits.head()


# In[7]:


target_fruits_name = dict(zip(fruits.fruit_label.unique(),
fruits.fruit_name.unique()))
target_fruits_name


# In[8]:


x = fruits[['mass', 'width', 'height']]
y = fruits['fruit_label']


# In[9]:


type(x)


# In[10]:


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)


# In[11]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)


# In[12]:


knn.fit(x_train,y_train)


# In[13]:


knn.score(x_test,y_test)


# In[14]:


fruits.head()


# In[67]:


x_test.columns


# In[80]:


single = pd.DataFrame([[180,8.0,6.8]], columns=x_test.columns)


# In[81]:


fruit_prediction = knn.predict(single)
target_fruits_name[fruit_prediction[0]]


# In[119]:


get_ipython().system('pip install mlxtend')


# In[120]:


# Plotting the classification results
from mlxtend.plotting import plot_decision_regions


# In[121]:


# Plotting the decision boundary
plot_decision_regions(x_test.values, y_test.values, clf = knn, legend = 2)
plt.title("Decision boundary using KNN Classification (Test)")
plt.xlabel("mean_perimeter")
plt.ylabel("mean_texture")


# In[135]:


from sklearn.decomposition import PCA
pca = PCA(n_components = 2)

x_train2 = pca.fit_transform(x_train)
knn.fit(x_train2, y_train)
plot_decision_regions(x_train2, y_train.values, clf=knn, legend=2)


# In[ ]:





# In[ ]:





# In[ ]:





# In[82]:


from adspy_shared_utilities import plot_two_class_knn
#plot_two_class_knn(x_train, y_train, 1, ‘uniform’, x_test, y_test)


# In[87]:


x_test.head()


# In[88]:


x_train.values


# In[97]:


x_test.values.shape


# In[102]:


plot_two_class_knn(x_train.values, y_train.values, 5, 'uniform', x_test.values, y_test.values)


# In[110]:


from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy




X_mat = x_train.values
y_mat = y_train.values

# Create color maps
cmap_light = ListedColormap(['#FFFFAA', '#AAFFAA', '#AAAAFF','#EFEFEF'])
cmap_bold  = ListedColormap(['#FFFF00', '#00FF00', '#0000FF','#000000'])

clf = KNeighborsClassifier(n_neighbors = 5, weights='uniform')
from matplotlib.colors import ListedColormap, BoundaryNorm


clf.fit(X_mat, y_mat)

 
   # Plot the decision boundary by assigning a color in the color map
    # to each mesh point.

mesh_step_size = .01  # step size in the mesh
plot_symbol_size = 50

x_min, x_max = X_mat[:, 0].min() - 1, X_mat[:, 0].max() + 1
y_min, y_max = X_mat[:, 1].min() - 1, X_mat[:, 1].max() + 1
xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, mesh_step_size),numpy.arange(y_min, y_max, mesh_step_size))
#Z = clf.predict(numpy.c_[xx.ravel(), yy.ravel()])    

print(xx.shape)
print(yy.shape)


# In[111]:


Z = clf.predict(numpy.c_[xx.ravel(), yy.ravel()])    


# In[115]:


def knn_comparison(data, k):
 x = data[['X','Y']].values
 y = data['class'].astype(int).values
 clf = neighbors.KNeighborsClassifier(n_neighbors=k)
 clf.fit(x, y)
# Plotting decision region
 plot_decision_regions(x, y, clf=clf, legend=2)
# Adding axes annotations
 plt.xlabel('X')
 plt.ylabel('Y')
 plt.title('Knn with K='+ str(k))
 plt.show()


# In[116]:


path = '/Users/Kaemyuijang/SCMA248/Data/Complete-KNN-visualization-master/ushape.csv'

data1 = pd.read_csv(path)
for i in [1,5,20,30,40,80]:
    knn_comparison(data1, i)


# In[ ]:





# In[ ]:


# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# scikit-learn modules
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Plotting the classification results
from mlxtend.plotting import plot_decision_regions

# Importing the dataset
dataset = load_breast_cancer() 

# Converting to pandas DataFrame
df = pd.DataFrame(dataset.data, columns = dataset.feature_names)
df['target'] = pd.Series(dataset.target)

print("Total samples in our dataset is: {}".format(df.shape[0]))

# Describe the dataset
df.describe()

# Selecting the features
features = ['mean perimeter', 'mean texture']
x = df[features]

# Target variable
y = df['target']

# Splitting the dataset into the training and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 25 )

# Fitting KNN Classifier to the Training set
model = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
model.fit(x_train, y_train)

# Predicting the results
y_pred = model.predict(x_test)

# Confusion matrix
print("Confusion Matrix")
matrix = confusion_matrix(y_test, y_pred)
print(matrix)

# Classification Report
print("\nClassification Report")
report = classification_report(y_test, y_pred)
print(report)

# Accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('KNN Classification Accuracy of the model: {:.2f}%'.format(accuracy*100))

# Plotting the decision boundary
plot_decision_regions(x_test.values, y_test.values, clf = model, legend = 2)
plt.title("Decision boundary using KNN Classification (Test)")
plt.xlabel("mean_perimeter")
plt.ylabel("mean_texture")


# In[ ]:




