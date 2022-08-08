# LIBRARIES

library(dplyr)
library(pROC)
library(mlbench)
library(caret)
library(e1071)
library(lime)
library(corrplot)
library(xgboost)
library(tictoc)

################################################################################
################################################################################

# DATA PREPARATION

# Read data
train <- read.csv("https://raw.githubusercontent.com/rohang2504/datasets/main/pom_681_proj/train.csv")

# Descriptive data analysis
summary(train)
str(train)

# Exploring dataset
corrplot(cor(train), method = "color")

# After dropping less significant variables and variables that might cause high skewness
# From the original correlation plot, ID seems to have a high correlation with Score, we will study this later on
trainset <- train[,c(1,2,3,13,14,15,16,17,18,21)]

# Correlation Plot of Data
corrplot(cor(trainset), method='color')

# Building training and testing sets
# Dataset without ID (only relevant medical data)
trainset_med <- trainset[1:3105,-c(1)]
testset_med <- trainset[3106:4141,-c(1)]

# Dataset with ID
trainset_id <- trainset[1:3105,]
testset_id <- trainset[3106:4141,]

################################################################################
################################################################################

# LINEAR REGRESSION

# Regression with relevant medical data
reg_med <- lm(score~., data = trainset_med)
summary(reg_med)

# Regression with ID included
reg_id <- lm(score~., data = trainset_id)
summary(reg_id)

# Preditions
reg_med_pred <- predict(reg_med, testset_med)
reg_id_pred <- predict(reg_id, testset_id)

# Studying RMSE
plot(reg_med_pred ~ testset_med$score, main = 'Expected v Observation - Test data')
reg_med_rmse <- sqrt(mean((reg_med_pred - testset_med$score)^2))
reg_med_rmse

plot(reg_id_pred ~ testset_id$score, main = 'Expected v Observation - Test data with ID')
reg_id_rmse <- sqrt(mean((reg_id_pred - testset_id$score)^2))
reg_id_rmse

# We get a lower RMSE with ID included in the dataset, which should not be the case

################################################################################
################################################################################

# RANDOM FORESTS

# Cross Validation parameters for RF
cv <- trainControl(method="adaptive_cv", 
                   number = 3, 
                   repeats = 2,
                   adaptive = list(min=2,
                                   alpha=0.03,
                                   method='gls',
                                   complete=T),
                   allowParallel = T,
                   verboseIter = T,
                   returnData = F,
                   search = 'random') 

# RF with relevant medical data
rf_med <- train(score ~ ., 
                data=trainset_med,
                method="rf",
                trControl=cv,
                importance=TRUE)
summary(rf_med)

# RF with ID included
rf_id <- train(score ~ ., 
                data=trainset_id,
                method="rf",
                trControl=cv,
                importance=TRUE)
summary(rf_id)

# Predictions
rf_med_pred <- predict(rf_med, testset_med)
rf_id_pred <- predict(rf_id, testset_id)

# Studying RMSE
plot(rf_med_pred ~ testset_med$score, main = 'Expected v Observation - Test data')
rf_med_rmse <- sqrt(mean((rf_med_pred - testset_med$score)^2))
rf_med_rmse

plot(rf_id_pred ~ testset_id$score, main = 'Expected v Observation - Test data with ID')
rf_id_rmse <- sqrt(mean((rf_id_pred - testset_id$score)^2))
rf_id_rmse

################################################################################
################################################################################

#XGBOOST

modelLookup("xgbTree")

# Cross Validation parameters for XGBoost
cv <- trainControl(method="adaptive_cv", 
                   number = 3, 
                   repeats = 2,
                   adaptive = list(min=2,
                                   alpha=0.03,
                                   method='gls',
                                   complete=T),
                   allowParallel = T,
                   verboseIter = T,
                   returnData = F,
                   search = 'random') 

# XGB with relevant medical data
xg_med <- train(score ~ ., 
             data = trainset_med,  
             trControl = cv,
             method = "xgbTree",
             tuneLength = 500) 
summary(xg_med)

# XGB with ID included in dataset
xg_id <- train(score ~ ., 
             data = trainset_id,  
             trControl = cv,
             method = "xgbTree",
             tuneLength = 500) 
summary(xg_id)

# Predictions
xg_med_pred <- predict(xg_med, testset_med)
xg_id_pred <- predict(xg_id, testset_id)

# Studying RMSE
plot(xg_med_pred ~ testset_med$score, main = 'Expected v Observation - Test data')
xg_med_rmse <- sqrt(mean((xg_med_pred - testset_med$score)^2))
xg_med_rmse

plot(xg_id_pred ~ testset_id$score, main = 'Expected v Observation - Test data with ID')
xg_id_rmse <- sqrt(mean((xg_id_pred - testset_id$score)^2))
xg_id_rmse


################################################################################
################################################################################

# TESTING WEB TEST DATA

testweb <- read.csv("https://raw.githubusercontent.com/rohang2504/datasets/main/pom_681_proj/test.csv")

# Predictions
rf_med_pred <- predict(rf_med, testweb)
rf_id_pred <- predict(rf_id, testweb)

# Submission
submission_id <- data.frame(Id = testweb$Id, Expected = rf_id_pred)
submission_med <- data.frame(Id = testweb$Id, Expected = rf_med_pred)


