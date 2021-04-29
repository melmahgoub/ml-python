#!/usr/bin/env python
# coding: utf-8

# ## Question 1

# ### A 

# I used the command cp /public/bmort/R/wheat-unknown.csv . and cp /public/bmort/R/wheat.csv .

# ### B

# In[1]:


data1<- read.csv("wheat.csv")


# In[2]:


colSums(is.na(data1))


# There are no missing Values

# ### C

# In[3]:


summary(data1)


# The ranges of the values differ mostly. Groove, width and length have a similar range. Area and perimeter also have a similar range of values. All of the values have similar magnitude, as in each column the values are within the same powers of 10.

# ### D

# In[4]:


library(class)
library(caret)


# In[5]:


n<- 100
sample_rate<- 0.8
ntest<- n*(1-sample_rate)
trainingrows<- sample(1:n, sample_rate*n, replace=F)
trainingrows
X_train<- subset(data1[trainingrows,], select=c(area, perimeter, compactness, length, width, asymmetry, groove))
testingrows<- setdiff(1:n, trainingrows)
testingrows
X_test<-  subset(data1[testingrows,], select=c(area, perimeter, compactness, length, width, asymmetry, groove))
X_test
y_train<- data1[trainingrows, ]$type
y_train
y_test<- data1[testingrows,]$type
y_test


# In[6]:


train_rows<- createDataPartition(y=data1$type, p=0.8, list=F)
training<- data1[train_rows,]
head(training)
testing<- data1[-train_rows,]
head(testing)


# ### E

# In[7]:


library(caret)


# In[8]:


trctrl<- trainControl(method="repeatedcv", number=10, repeats=3)
svm_linear<- train(type~., data=training, method="svmLinear", trControl=trctrl,PreProcess=c("center", "scale"))


# In[9]:


svm_linear


# ### F

# In[10]:


test_pred<- predict(svm_linear, newdata=testing)
test_pred


# In[11]:


confusionMatrix(test_pred, testing$type)


# The accuracy of this test is 94.74%, which is generally good. However, since the confidence interval is so wide, (82.25%, 99.36%), this result is not that reliable.

# ## G

# In[12]:


testdata<- read.csv("wheat-unknown.csv")
testfit<- predict(svm_linear, newdata=testdata)
testfit


# In[13]:


testdata$type<- testfit


# In[14]:


testdata


# In[ ]:




