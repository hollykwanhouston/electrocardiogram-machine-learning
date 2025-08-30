# svm_features.R
# SVM (RBF) on FeaturesDatasets
install.packages("e1071", repos='https://cloud.r-project.org')  # if not already installed
install.packages("readr")
install.packages("class")
install.packages("dplyr")

library(e1071)
library(readr)
library(class)
library(dplyr)

train_feat <- read.csv("train_feat.csv")
# Select feature columns (adjust indices if needed)
train_feat <- train_feat[, 2:190]
train_feat$Type <- as.factor(train_feat$Type)

set.seed(50)
s <- floor(0.65 * nrow(train_feat))
set.seed(20)
train_feat_index <- sample(seq_len(nrow(train_feat)), size = s)
train <- train_feat[train_feat_index, ]
test <- train_feat[-train_feat_index, ]

# Initial SVM fit and tuning
svmfit <- svm(Type ~ ., data = train, kernel = "radial", scale = FALSE)
summary(svmfit)

svm_tune <- tune.svm(Type ~ ., data = train, kernel = "radial", gamma = 2^(-1:1), cost = 2^(2:4), scale = FALSE)
print(svm_tune)

# Final model with selected parameters
svmfit <- svm(Type ~ ., data = train, kernel = "radial", cost = 8, gamma = 0.5, scale = FALSE)
summary(svmfit)

# Predictions and confusion table
svm.pred <- predict(svmfit, test)
table1 <- table(svm.pred, test$Type)
print(table1)
