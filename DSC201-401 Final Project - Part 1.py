#!/usr/bin/env python
# coding: utf-8

data1<- read.csv("wheat.csv")

colSums(is.na(data1)) # There are no missing Values

summary(data1) # The ranges of the values differ mostly.

'''
Groove, width and length have a similar range. Area and perimeter also have a similar range of values. 
All of the values have similar magnitude, as in each column the values are within the same powers of 10.
'''

library(class)
library(caret)

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


train_rows<- createDataPartition(y=data1$type, p=0.8, list=F)
training<- data1[train_rows,]
head(training)
testing<- data1[-train_rows,]
head(testing)


library(caret)

trctrl<- trainControl(method="repeatedcv", number=10, repeats=3)
svm_linear<- train(type~., data=training, method="svmLinear", trControl=trctrl,PreProcess=c("center", "scale"))

svm_linear


test_pred<- predict(svm_linear, newdata=testing)
test_pred


confusionMatrix(test_pred, testing$type)


'''
The accuracy of this test is 94.74%, which is generally good. However, since the confidence interval is so wide, (82.25%, 99.36%), this result is not that reliable.
'''


testdata<- read.csv("wheat-unknown.csv")
testfit<- predict(svm_linear, newdata=testdata)
testfit



testdata$type<- testfit

testdata
