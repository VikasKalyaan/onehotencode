#!/usr/bin/env python
# coding: utf-8

# In[33]:


import numpy as np
import pandas as pd


# In[34]:


df = pd.read_csv("G://car_price.csv")


# In[35]:


df.head()


# In[36]:


df['fuel_type'].value_counts()


# In[37]:


pd.get_dummies(df,columns=['fuel_type','ownership'],drop_first=True)


# In[50]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(df.iloc[:,1:7],df.iloc[:,-1],test_size = 0.2,random_state=2)


# In[51]:


X_test.head()


# In[52]:


from sklearn.preprocessing import OneHotEncoder


# In[60]:


ohe = OneHotEncoder()


# In[63]:


X_train_new=ohe.fit_transform(X_train[['fuel_type','ownership']]).toarray()


# In[66]:


X_test_new = ohe.transform(X_test[['fuel_type','ownership']]).toarray()


# In[67]:


X_train_new


# In[78]:


np.hstack((X_train[['car_name','kms_driven']].values,X_train_new))


# In[71]:


counts = df['car_name'].value_counts()


# In[72]:


df['car_name'].unique()
threshold = 100


# In[76]:


repl =  counts[counts <= threshold].index


# In[77]:


pd.get_dummies(df['car_name'].replace(repl, 'uncommon'))


# In[ ]:




