library(dplyr)
library(pROC)
library(mlbench)
library(caret)
library(e1071)
library(lime)
library(corrplot)

# Read data
data <- read.csv("https://raw.githubusercontent.com/rohang2504/datasets/main/pom_681_proj/train.csv")

# Descriptive data analysis

summary(data)

# Pair Plot of data with score
pairs(~score+Id+V3+V4+V5+V6+V7, data=data)
pairs(~score+V8+V9+V10+V11+V12, data=data)
pairs(~score+V13+V14+V15+V16+V17, data=data)
pairs(~score+V18+V19+V20+V21+V22, data=data)
pairs(~score+V23+V24+V25+V26+V27+V28+V29, data=data)

# Correlation Plot of Data
cor_mat <- cor(data)
corrplot(cor_mat, method='color')

# It seems from the correlation plot that V3-V& are highly correlated and same for V8-V13
# This has the potential to skew our analysis

data <- data[,c(1,2,3,13,14,15,16,17,18,21)]

# Correlation Plot of Data
cor_mat <- cor(mydata2)
corrplot(cor_mat, method='color')

# Medical Condition Data
data2 <- data[,-c(1)]
mydata <- as.data.frame((data2))
set.seed(1234) 
ind <- sample(2, nrow(mydata), replace = T, prob = c(0.8, 0.2))
train <- mydata[ind == 1,]
test <- mydata[ind == 2,]

# Keeping ID column
mydata2 <- data
set.seed(1234) 
ind <- sample(2, nrow(mydata2), replace = T, prob = c(0.8, 0.2))
train <- mydata2[ind == 1,]
test <- mydata2[ind == 2,]

###############################################################################################

# Regression

# Simple Linear Regression

# Training
mod <- lm(score~Id, data = train2)
summary(mod)

# Testing
test_reg <- predict(mod, test)
plot(test_reg ~ test$score, main = 'Predicted Vs Actual Score - Test data')
sqrt(mean((test$score - test_reg)^2))
cor(test$score, test_reg) ^2

###############################################################################################

# RF
set.seed(1234)
# train <- train[,-c(3,10)]
cor_mat <- cor(train_sig)
corrplot(cor_mat, method='color')


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

# Dropping least important variable
train2 <- train[,-c(10)]

forest <- train(score ~ Id, 
                data=train2,
                method="rf",
                trControl=cv,
                importance=TRUE)
plot(varImp(forest))

# Plot, RMSE, R-square
rf <-  predict(forest,  test)
plot(rf ~ test$score, main = 'Predicted Vs Actual MEDV - Test data')
sqrt(mean((test$score - rf)^2))
cor(test$score, rf) ^2

# Explain predictions
explainer <- lime(test[1:3,], forest, n_bins = 5)
explanation <- explain( x = test[1:3,], 
                        explainer = explainer, 
                        n_features = 5)
plot_features(explanation)

###############################################################################################

# XGBoost Adaptive search
library(xgboost) 
library(caret)
library(tictoc)

modelLookup("xgbTree")
set.seed(1234) 
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
set.seed(1234) 
boo <- train(score ~ Id, 
             data = train,  
             trControl = cv,
             method = "xgbTree",
             tuneLength = 500) 

plot(varImp(boo))

# Plot, RMSE, R-square
bo <-  predict(boo,test)
plot(bo ~ test$score, main = 'Predicted Vs Actual Score - Test data')
sqrt(mean((test$score - bo)^2))


###############################################################################################

# XGBoost Adaptive search - Fine tuning
library(xgboost) 
library(caret)
library(tictoc)

modelLookup("xgbTree")
set.seed(1234) 
cv <- trainControl(method="adaptive_cv", 
                   number = 3, 
                   repeats = 2,
                   adaptive = list(min=2,
                                   alpha=0.03,
                                   method='gls',
                                   complete=T),
                   allowParallel = T,
                   verboseIter = T,
                   returnData = F)

g <- expand.grid(nrounds = seq(from = 650, to=850, by=5),
                 max_depth = 10,
                 eta = seq(from = 0.130, to=0.140, by=0.001),
                 gamma = 3.59,
                 colsample_bytree = 0.408,
                 min_child_weight = 6,
                 subsample = 0.519)
set.seed(1234) 
boo <- train(score ~Id, 
             data = train,  
             trControl = cv,
             method = "xgbTree",
             tuneGrid = g) 

plot(varImp(boo))

# Plot, RMSE, R-square
bo <-  predict(boo,test)
plot(bo ~ test$score, main = 'Predicted Vs Actual Score - Test data')
sqrt(mean((test$score - bo)^2))
cor(test$score, rf) ^2

#########################################################################################

# The code was repeated for the three different variations on the dataset. 
# Only thing changed was the formula and the training and testing data.

#########################################################################################
# Test data from website
test_final <- read.csv(file.choose(), header = T)

# After removing correlated variables
test_final_sig <- test_final[,c(1,2,3,12,13,14,15,16,17,20)]

# Making Predictions
bo <-  predict(boo,  test_final_sig)

submission <- data.frame(test_final$Id, bo)
colnames(submission) <- c('Id', 'Expected')

setwd("~/Desktop")
write.csv(submission, "submission.csv",row.names = F)