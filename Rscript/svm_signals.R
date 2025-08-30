# svm_signals.R
# SVM (RBF) on SignalsDatasets
install.packages("e1071", repos='https://cloud.r-project.org')  # if not already installed
install.packages("readr")
install.packages("class")
install.packages("dplyr")

library(e1071)
library(readr)
library(class)
library(dplyr)

train_signal <- read.csv("train_signal.csv")
# Select feature columns (adjust indices if needed)
train_signal <- train_signal[, 2:6002]
train_signal$Type <- as.factor(train_signal$Type)

set.seed(50)
s <- floor(0.65 * nrow(train_signal))
train_signal_index <- sample(seq_len(nrow(train_signal)), size = s)
train <- train_signal[train_signal_index, ]
test <- train_signal[-train_signal_index, ]

svmfit <- svm(Type ~ ., data = train, kernel = "radial", scale = FALSE)
summary(svmfit)

svm_tune <- tune.svm(Type ~ ., data = train, kernel = "radial", gamma = seq(1/2^nrow(train_feat), 1, 0.01), cost = 2^seq(-6, 4, 2), scale = FALSE)
print(svm_tune)

svmfit <- svm(Type ~ ., data = train, kernel = "radial", cost = 4, gamma = 0.5, scale = FALSE)
summary(svmfit)

svm.pred <- predict(svmfit, test)
table1 <- table(svm.pred, test$Type)
print(table1)
