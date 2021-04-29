#!/usr/bin/env python
# coding: utf-8

# ## Question 2

# ### A

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv("/public/bmort/python/bank.csv")


# ## B

# In[3]:


df.isna().sum()


# In[4]:


imputed_value= df['age'].mean()
df['age'] = df['age'].fillna(imputed_value)
df.isna().sum()


# There was 1 missing value of age so I imputed it with the mean value of the age.

# ## C

# In[5]:


df.describe()


# In[6]:


df


# Previous and campaign have a similar range values. emp.var.rate and euribo3m have similar range values also. Age and cons.conf.indx have similar range values. Values for age, previous, cons.price.idx, cons.conf.idx and nr.employed have the similar magnitude in each column, as in each column the values are within the same powers of 10.

# ## D

# In[7]:


from sklearn import preprocessing


# In[8]:


le = preprocessing.LabelEncoder()


# In[9]:


le.fit(df['job'])
df['job'] = le.transform(df['job'])
le.fit(df['marital'])
df['marital'] = le.transform(df['marital'])
le.fit(df['education'])
df['education'] = le.transform(df['education'])
le.fit(df['default'])
df['default'] = le.transform(df['default'])
le.fit(df['housing'])
df['housing'] = le.transform(df['housing'])
le.fit(df['loan'])
df['loan'] = le.transform(df['loan'])
le.fit(df['contact'])
df['contact'] = le.transform(df['contact'])
le.fit(df['month'])
df['month'] = le.transform(df['month'])
le.fit(df['poutcome'])
df['poutcome'] = le.transform(df['poutcome'])
le.fit(df['day_of_week'])
df['day_of_week'] = le.transform(df['day_of_week'])
le.fit(df['y'])
df['y'] = le.transform(df['y'])
list(df.columns)


# ## E & F

# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


X = df[['age',
 'job',
 'marital',
 'education',
 'default',
 'housing',
 'loan',
 'campaign',
 'previous',
 'poutcome',
 'emp.var.rate',
 'cons.price.idx',
 'cons.conf.idx',
 'euribor3m',
 'nr.employed']].to_numpy()


# In[12]:


y =df['y'].to_numpy()


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)


# ## G

# In[14]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver = 'liblinear')
model.fit(X_train, y_train)


# ## H

# In[15]:


from sklearn.model_selection import KFold, cross_val_score
kfold= KFold(n_splits =5, shuffle=True)
scores = cross_val_score(model, X_train, y_train, cv=kfold)
scores
print("Accuracy: %0.2f +/- %0.2f" % (scores.mean(), scores.std()) )


# The accuracy of the model based on KFold cross validation is 90%

# ## I

# In[16]:


from sklearn import metrics
y_pred=model.predict(X_test)
metrics.accuracy_score(y_test, y_pred)


# ## J

# In[17]:



cmatrix = metrics.confusion_matrix(y_test, y_pred)
cmatrix


# In[18]:


metrics.accuracy_score(y_test, y_pred)


# The accuracy of the model on the test set is slightly lower than the one on the training set.

# ## K

# In[19]:


testdf= pd.read_csv('/public/bmort/python/bank-unknown.csv')
testdf1= pd.read_csv('/public/bmort/python/bank-unknown.csv')
testdf


# In[20]:


le.fit(testdf['job'])
testdf['job'] = le.transform(testdf['job'])
le.fit(testdf['marital'])
testdf['marital'] = le.transform(testdf['marital'])
le.fit(testdf['education'])
testdf['education'] = le.transform(testdf['education'])
le.fit(testdf['default'])
testdf['default'] = le.transform(testdf['default'])
le.fit(testdf['housing'])
testdf['housing'] = le.transform(testdf['housing'])
le.fit(testdf['loan'])
testdf['loan'] = le.transform(testdf['loan'])
le.fit(testdf['contact'])
testdf['contact'] = le.transform(testdf['contact'])
le.fit(testdf['month'])
testdf['month'] = le.transform(testdf['month'])
le.fit(testdf['poutcome'])
testdf['poutcome'] = le.transform(testdf['poutcome'])
le.fit(testdf['day_of_week'])
testdf['day_of_week'] = le.transform(testdf['day_of_week'])


list(testdf.columns)


# In[21]:


Xtest = testdf[['age',
 'job',
 'marital',
 'education',
 'default',
 'housing',
 'loan',
 'campaign',
 'previous',
 'poutcome',
 'emp.var.rate',
 'cons.price.idx',
 'cons.conf.idx',
 'euribor3m',
 'nr.employed']].to_numpy()


# In[22]:


ytest=model.predict(Xtest)


# In[23]:


testdf['y']=ytest


# In[24]:


testdf[testdf['y']==1]


# ## L

# In[25]:


#campaign	pdays	previous	emp.var.rate	cons.price.idx	cons.conf.idx	euribor3m	nr.employed
import numpy as np
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
testdf1[['age']]= scaler.fit_transform(testdf1[['age']]) ## scale data in the age column
testdf1[['campaign']]= scaler.fit_transform(testdf1[['campaign']])
testdf1[['pdays']]= scaler.fit_transform(testdf1[['pdays']])
testdf1[['previous']]= scaler.fit_transform(testdf1[['previous']])
testdf1[['emp.var.rate']]= scaler.fit_transform(testdf1[['emp.var.rate']])
testdf1[['cons.price.idx']]= scaler.fit_transform(testdf1[['cons.price.idx']])
testdf1[['cons.conf.idx']]= scaler.fit_transform(testdf1[['cons.conf.idx']])
testdf1[['euribor3m']]= scaler.fit_transform(testdf1[['euribor3m']])
testdf1[['nr.employed']]= scaler.fit_transform(testdf1[['nr.employed']])
testdf1.fillna(testdf1.mean())


# In[26]:


le.fit(testdf1['job'])
testdf1['job'] = le.transform(testdf1['job'])
le.fit(testdf1['marital'])
testdf1['marital'] = le.transform(testdf1['marital'])
le.fit(testdf1['education'])
testdf1['education'] = le.transform(testdf1['education'])
le.fit(testdf1['default'])
testdf1['default'] = le.transform(testdf1['default'])
le.fit(testdf1['housing'])
testdf1['housing'] = le.transform(testdf1['housing'])
le.fit(testdf1['loan'])
testdf1['loan'] = le.transform(testdf1['loan'])
le.fit(testdf1['contact'])
testdf1['contact'] = le.transform(testdf1['contact'])
le.fit(testdf1['month'])
testdf1['month'] = le.transform(testdf1['month'])
le.fit(testdf1['poutcome'])
testdf1['poutcome'] = le.transform(testdf1['poutcome'])
le.fit(testdf1['day_of_week'])
testdf1['day_of_week'] = le.transform(testdf1['day_of_week'])


# In[27]:


Xx= testdf1[['age',
 'job',
 'marital',
 'education',
 'default',
 'housing',
 'loan',
 'campaign',
 'previous',
 'poutcome',
 'emp.var.rate',
 'cons.price.idx',
 'cons.conf.idx',
 'euribor3m',
 'nr.employed']].to_numpy()


# In[28]:


y1test=model.predict(Xx)


# In[29]:


metrics.accuracy_score(ytest, y1test)


# In[30]:


testdf1['y']=y1test


# In[31]:


testdf1[testdf1['y']==1]


# The accuracy is significantly lower and the labels assigned to the unknown customers do change. 

# In[ ]:




