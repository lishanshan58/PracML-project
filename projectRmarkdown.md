# Final Report for Practical Machine Learning Course Project
### Practical Machine Learning @ Coursera.org


The use of modern devices has made it relatively inexpensive to collect a large amount of data about personal activity. We may use such data to quantify not only how much of a particular activity people do, but how well they do it as well. In this project, a large set of data collected from accelerometers on the belt, forearm, and dumbbell of 6 participants will be used to predict the manner in which they did the exercise. 

The training set contains 19,622 observations of 160 variables, each observation being labeled as one of 5 levels in terms of how exactly according to the specification was the weight lifting exercises finished. In view of the sheer size of the dataset, a preliminary screening process of the predictors was performed to exclude any variables that have missing values, leaving 93 variables. Then a check for near zero variables was performed to exclude variables with little variability, along with the ones for observation number, participant names, and timestamp. After the two-step pre-processing, I was left with 54 potential variables to build a prediction model.

In order to decide which machine learning technique should be used for building the model, firstly a very small portion (10%) of the training data was extracted, with which 3 different machine learning methods were performed to generate a predictive model, namely recursive partitioning and regression trees (R: {rpart}), random forest (R: {randomForest}), and gradient boosting method (R: {gbm}). When applying these three models to the remaining data (90% of the entire training set) for prediction, the misclassification rates were quite different: 97.22% for rpart, 2.25% for random forest, and 3.24% for boosting. Obviously, the latter two ensemble methods worked much better than the simpler one on this classification problem. In consideration of the relative time inefficiency of boosting method, I chose random forest to build my final predictive model.     

Next, a 4-fold cross validation was performed to evaluate the out of sample error. To avoid the issue of memory overflow often encountered when dealing with super large data set in R, I used 1/4 of the data as training set and the remaining 3/4 as validation set in the cross validation stage. The model trained from training set and the prediction accuracy obtained on validation set were recorded for each of the 4 cross validation steps, generating an average of 98.74% correctly classified instances, equivalently 1.26% out of sample error.

Finally, the model with the highest prediction accuracy rate on validation set (99.06%) was chosen as the final model to predict on the testing set, which produced a final misclassification rate of .69%. 

To conclude, I would expect the out of sample error to be around 1%, given the above evidence. Further tests will need to be done to estimate the bias and variance embedded in the expected prediction error with more accuracy.




**Reference:**

Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.




```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
pmldata = read.csv("E:/practicalMachineLearning/pml-training.csv", header = T)

# OMIT VARIABLES WITH MISSING VALUE
na.var = apply(pmldata, 2, function(v){sum(is.na(v))})
traindata = pmldata[,na.var==0]

# CHECK FOR NEAR ZERO VARIABLES
check = nearZeroVar(traindata)  # return 34 problematic variables
check = c(1:5, check) # further exclude non-informative variables
traindata = traindata[, -check]

# Extract 20% of data to decide machine learning technique to be used
inTrain = createDataPartition(y = traindata$classe, p=.15, list = F)
train1 = traindata[inTrain,]
test1 = traindata[-inTrain,]


# TRY THREE DIFFERENCE MACHINE LEARNING METHODS

# RPART
library(rpart)
mod1 = rpart(classe~., data = train1)
pred1 = predict(mod1, newdata = test1 )
pred1v=apply(pred1, 1, function(row){c('A','B','C','D','E')[row==1]})
mod1acc = sum(pred1v == (test1$classe))/ length(test1$classe)
1-mod1acc 
```

```
## [1] 0.9853082
```

```r
# RANDOM FOREST
library(randomForest)
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
mod2 = randomForest(classe~., data = train1)
pred2 = predict(mod2, newdata = test1) # return a list of factors
mod2acc = sum(pred2 == (test1$classe))/ length(test1$classe)
1-mod2acc 
```

```
## [1] 0.02698489
```

```r
# BOOSTING
library(gbm)
```

```
## Loading required package: survival
## Loading required package: splines
## 
## Attaching package: 'survival'
## 
## The following object is masked from 'package:caret':
## 
##     cluster
## 
## Loading required package: parallel
## Loaded gbm 2.1
```

```r
mod3v = gbm(classe~.,data=train1,n.trees = 150, shrinkage = 0.1,interaction.depth = 3)
```

```
## Distribution not specified, assuming multinomial ...
```

```r
pred3v = predict(mod3v, newdata = test1,n.trees=150)
pred3v = pred3v[,,1]
pred3v=apply(pred3v, 1, function(row){c('A','B','C','D','E')[which(row == max(row))]})
mod3acc=sum(pred3v == (test1$classe))/ length(test1$classe)
1-mod3acc  
```

```
## [1] 0.03789878
```

```r
# CROSS VALIDATION

rf.cv = function(){
  
  folds = createFolds(y = traindata$classe, k = 4, returnTrain=T)
  
  results = list()
  for (i in 1:4){
    test0 = traindata[folds[[i]], ]
    train0 = traindata[-folds[[i]],]
    
    mod0 = randomForest(classe~., data = train0)
    pred0 = predict(mod0, newdata = test0 )
    accuracy = sum(pred0 == (test0$classe))/ length(test0$classe)
    
    print(accuracy)
    results[[i]] = list(model = mod0, accuracy = accuracy) 
    
  } 
  return(results)
}
results = rf.cv()
```

```
## [1] 0.9872248
## [1] 0.9900122
## [1] 0.9864094
## [1] 0.986953
```

```r
# USE 2ND AS THE FINAL MODEL
finalmod = results[[2]]$model

pmltest = read.csv("E:/practicalMachineLearning/pml-testing.csv", header = T)
finalpmltest = predict(finalmod, newdata = pmltest)
finalpmltest
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```
