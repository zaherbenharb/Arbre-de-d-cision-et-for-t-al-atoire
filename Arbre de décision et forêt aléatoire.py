#!/usr/bin/env python
# coding: utf-8

# In[271]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree   
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import graphviz 
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics


# In[272]:


data=pd.read_csv("titanic-passengers.csv",sep=";")


# In[273]:


data.head()


# In[274]:


data.drop(['Ticket','Cabin','PassengerId','Name'],axis=1,inplace=True)


# In[275]:


data.head()


# In[276]:


data['Age'].fillna(data['Age'].mean(),inplace=True)


# In[ ]:





# In[277]:


data.head()


# In[278]:


data['Sex'].replace(['male','female'],[0,1],inplace=True)
data['Survived'].replace(['No','Yes'],[0,1],inplace=True)
data['Embarked'].replace(['S','C','Q'],[1,2,3],inplace=True)


# In[279]:


data.head()


# In[280]:


data.describe()


# In[281]:


x= data[['Pclass','Sex','Age','SibSp','Parch']]
y= data['Survived']


# In[ ]:





# In[282]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.20,random_state=10)
tree = tree.DecisionTreeClassifier(random_state=10)  
tree.fit(x_train, y_train)   
y_pred=tree.predict(x_test)  
print("score:{}".format(accuracy_score(y_test, y_pred)))


# In[ ]:





# In[283]:


from sklearn import tree
x= data[['Pclass','Sex','Age']]
y= data['Survived']
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.20,random_state=10)
model = tree.DecisionTreeClassifier()  
model.fit(x_train, y_train)   
y_pred=model.predict(x_test)  
print("score:{}".format(accuracy_score(y_test, y_pred)))


# In[284]:



dot_data = tree.export_graphviz(model, out_file=None)
graph = graphviz.source(dot_data)
graph.render("data")
graph


# In[285]:


x= data[['Pclass','Sex','Age']]
y= data['Survived']
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.20,random_state=10)
dmodel = tree.DecisionTreeClassifier(criterion = "gini", splitter='random',max_leaf_nodes=10,min_samples_leaf = 5, max_depth=5)  
dmodel.fit(x_train, y_train)   
y_pred=dmodel.predict(x_test)  
print("score:{}".format(accuracy_score(y_test, y_pred)))


# # Prédiction de forêt aléatoire
# 

# In[286]:


clf=RandomForestClassifier(n_estimators=10)  #Creating a random forest with 100 decision trees
clf.fit(x_train, y_train)  #Training our model
y_pred=clf.predict(x_test)  #testing our model
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




