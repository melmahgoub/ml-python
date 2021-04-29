#!/usr/bin/env python
# coding: utf-8
# Jupyter notebook with 1 core and 4 GB of RAM using the Python 3 anaconda3 2019.3 kernel

'''
Summary Statistics & Data Pre-processing
'''

import pandas as pd


df=pd.read_csv("/public/melmahgo/python/bank.csv")
df.isna().sum()


imputed_value= df['age'].mean()
df['age'] = df['age'].fillna(imputed_value) # impute missing value with mean value of the age.
df.isna().sum() 

df.describe()
df


from sklearn import preprocessing

le = preprocessing.LabelEncoder()

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


'''
Partition the data set so that 80% is used for training and 20% used for
testing the model
'''


from sklearn.model_selection import train_test_split



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




y =df['y'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

'''
Using Logistic regression model as the outcome is binary
'''


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver = 'liblinear')
model.fit(X_train, y_train)


'''
Using Scikit-Learn's KFold() k-fold cross-validation function on the training data to
demonstrate that the model does not overfit the data. 
'''


from sklearn.model_selection import KFold, cross_val_score
kfold= KFold(n_splits =5, shuffle=True)
scores = cross_val_score(model, X_train, y_train, cv=kfold)
scores
print("Accuracy: %0.2f +/- %0.2f" % (scores.mean(), scores.std()) )


# The accuracy of the model based on KFold cross validation is 90%

'''
Testing model on predicting the action of the customer to sign up or not sign up for a checking account.
'''


from sklearn import metrics
y_pred=model.predict(X_test)
metrics.accuracy_score(y_test, y_pred)


'''
Confusion matrix for the test data set to demonstrate the accuracy of the
model.
'''

cmatrix = metrics.confusion_matrix(y_test, y_pred)
cmatrix


metrics.accuracy_score(y_test, y_pred)


# The accuracy of the model on the test set is slightly lower than the one on the training set.

'''
Classifying the outcome of the customers on new data, based on model
'''


testdf= pd.read_csv('/public/bmort/python/bank-unknown.csv')
testdf1= pd.read_csv('/public/bmort/python/bank-unknown.csv')
testdf


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





ytest=model.predict(Xtest)
testdf['y']=ytest
testdf[testdf['y']==1]

'''
Complete the analysis once more, but this time preprocessing the data with
standardization applied to the columns containing numerical data.
'''


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

y1test=model.predict(Xx)
metrics.accuracy_score(ytest, y1test)
testdf1['y']=y1test
testdf1[testdf1['y']==1]


# The accuracy is significantly lower and the labels assigned to the unknown customers do change. 
# i.e. the additional preprocessing impacts the accuracy and outcome of the model




