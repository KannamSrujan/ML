#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df=pd.read_csv("diabetes.csv")


# In[2]:


df.head()


# In[3]:


df.shape


# In[4]:


df.isnull().sum()


# In[5]:


X=df.iloc[:,:-1].to_numpy()

y=df.iloc[:,-1].to_numpy()


# In[6]:


from sklearn.model_selection import train_test_split
'''
y-target variable
X-input columns
random_state=0 ensures that the data split will be the same every time you run the code with the same input data.
test size=20%
train size=80%
'''
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[7]:


from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier(random_state=0)
clf.fit(X_train,y_train)


# In[8]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.tree import plot_tree#This imports the plot_tree function from scikit-learn's tree module. 
#This function allows you to visualize decision trees trained using scikit-learn.

plt.figure(figsize=(20,10))#The figsize parameter specifies the width and height of the figure in inches.
'''feature_names parameter specifies the names of the features used in the tree, and 
the class_names parameter specifies the names of the target classes.'''
plot_tree(clf,feature_names=['Glucose','BMI'],class_names=['No','Yes'])
plt.show()


# In[9]:


clf.set_params(max_depth=3)


# In[10]:


clf.fit(X_train,y_train)
plt.figure(figsize=(20,10))
plot_tree(clf,feature_names=['Glucose','BMI'],class_names=['No','Yes'])
plt.show()


# In[11]:


predictions=clf.predict(X_test)


# In[12]:


clf.predict([[90,20],[200,30]])


# In[13]:


from sklearn.model_selection import cross_val_score
#cross_val_score function
'''
This line performs 5-fold cross-validation (cv=5) on your decision tree classifier (clf) using the training data (X_train and y_train). 
The scoring parameter specifies that you're interested in accuracy as the evaluation metric.'''
scores=cross_val_score(clf,X_train,y_train,cv=5,scoring='accuracy')
accuracy=scores.mean()
accuracy


# In[15]:


from sklearn import metrics 
cf=metrics.confusion_matrix(y_test,predictions)
cf


# In[16]:


tp=cf[1][1]
tn=cf[0][0]
fp=cf[0][1]
fn=cf[1][0]
print(f"tp:{tp}, tn:{tn},fp:{fp},fn:{fn}")


# In[17]:


print("accuracy",metrics.accuracy_score(y_test,predictions))


# In[18]:


print("Precision",metrics.precision_score(y_test,predictions))


# In[19]:


print("Recall",metrics.recall_score(y_test,predictions))


# In[21]:


feature_importances = clf.feature_importances_
print("Feature importances:",feature_importances)


# In[ ]:




