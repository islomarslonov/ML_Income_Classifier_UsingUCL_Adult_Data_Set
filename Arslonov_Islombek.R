##################################################
# ECON 418-518 Homework 3
# Islombek Arslonov
# The University of Arizona
# islomarslonov@arizona.edu 
# 8 December 2024
###################################################


#####################
# Preliminaries
#####################

# Clear environment, console, and plot pane
# rm(list = ls())
# cat("\014")
# graphics.off()

# Turn off scientific notation
# options(scipen = 999)

# Load packages
# pacman::p_load(data.table)

# Set sead
# set.seed(418518)

# library(dplyr)
# 
# df <- read.csv("../downloads/ECON_418-518_HW3_Data.csv")

#####################
# Problem 1
#####################


#################
# Question (i)
#################

# df <- df %>% select(-fnlwgt, -occupation, -relationship,
#                     -'capital.gain', -'capital.loss', -'educational.num')

#################
# Question (ii)
#################

##############
# Part (a)
##############

# df$income <- ifelse(df$income == ">50K", 1, 0)

##############
# Part (b)
##############
# df$race <- ifelse(df$race == "White", 1, 0)

##############
# Part (c)
##############
# df$gender <- ifelse(df$gender == "Male", 1, 0)

##############
# Part (d)
##############
# df$workclass <- ifelse(df$workclass == "Private", 1, 0)

##############
# Part (e)
##############
# df$native.country <- ifelse(df$native.country == "United-States", 1, 0)

##############
# Part (f)
##############
# df$marital.status <- ifelse(df$marital.status == "Married-civ-spouse", 1, 0)

##############
# Part (g)
##############
# df$education <- ifelse(df$education %in% c("Bachelors", "Masters", "Doctorate"), 1, 0)

##############
# Part (h)
##############
# df$age_sq <- df$age^2

##############
# Part (i)
##############

# standardize <- function(x) {
#   return((x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE))
# }

# df$age <- standardize(df$age)
# df$age_sq <- standardize(df$age_sq)
# df$hours.per.week <- standardize(df$hours.per.week)
# 
# head(df)


#################
# Question (iii)
#################

##############
# Part (a)
##############
# proportion_income <- mean(df$income == 1)
# cat("Proportion of individuals with income > $50,000:", proportion_income, "\n")

##############
# Part (b)
##############
# proportion_private <- mean(df$workclass == 1)
# cat("Proportion of individuals in the private sector:", proportion_private, "\n")

##############
# Part (c)
##############
# proportion_married <- mean(df$`marital.status` == 1)
# cat("Proportion of married individuals:", proportion_married, "\n")

##############
# Part (d)
##############
# proportion_females <- mean(df$gender == 0)
# cat("Proportion of females:", proportion_females, "\n")

##############
# Part (e)
##############
# total_observations <- sum(!is.na(df))
# cat("Total number of observations (non-NA values) in the dataset:", total_observations, "\n")
# total_na_values <- sum(is.na(df))
# cat("Total number of NA values in the dataset:", total_na_values, "\n")

##############
# Part (f)
##############
# df$income <- as.factor(df$income)
# cat("Income column converted to factor data type. Levels:", levels(df$income), "\n")
# 

#################
# Question (iv)
#################

##############
# Part (a)
##############
# n <- nrow(df) 
# last_train_index <- floor(n * 0.70)

##############
# Part (b)
##############

# training_data <- df[1:last_train_index, ]

##############
# Part (c)
##############

# testing_data <- df[(last_train_index + 1):n, ]
# 
# cat("Number of rows in the training set:", nrow(training_data), "\n")
# cat("Number of rows in the testing set:", nrow(testing_data), "\n")


#################
# Question (v)
#################

##############
# Part (a)
##############


# Lasso and ridge regression are useful when we want to avoid overfitting. They help by adding
# a penalty to the size of the coefficients, which makes the model simpler and more general.
# 
# Difference:
#   
# Lasso: It can shrink some coefficients to exactly zero, which means it selects only the most 
#   important variables.
# Ridge: It shrinks coefficients but does not make them exactly zero. Instead, it keeps all
#   variables but reduces their influence.
# 
# Pros and Cons:
#   
# Lasso Pros: Helps with feature selection.
# Lasso Cons: May struggle when variables are highly correlated.
# Ridge Pros: Handles correlated variables better and keeps all variables in the model.
# Ridge Cons: Does not perform feature selection, so all variables are included.


##############
# Part (b)
##############

# library(caret)
# library(glmnet)
# 
# lambda_grid <- 10^seq(2, -2, length = 50)
# 
# train_control <- trainControl(method = "cv", number = 10)
# 
# lasso_model <- train(
#   income ~ .,
#   data = training_data,
#   method = "glmnet",
#   trControl = train_control,
#   tuneGrid = expand.grid(alpha = 1, lambda = lambda_grid)
# )

##############
# Part (c)
##############

# best_lambda <- lasso_model$bestTune$lambda
# best_accuracy <- max(lasso_model$results$Accuracy)
# 
# cat("Best lambda:", best_lambda, "\n")
# cat("Best classification accuracy:", best_accuracy, "\n")

##############
# Part (d)
##############

# important_coefficients <- coef(lasso_model$finalModel, s = best_lambda)
# coeff_matrix <- as.matrix(important_coefficients)
# 
# non_zero_coefficients <- rownames(coeff_matrix)[coeff_matrix != 0]
# cat("Variables with non-zero coefficients:", non_zero_coefficients, "\n")


##############
# Part (e)
##############

# selected_vars <- non_zero_coefficients[!non_zero_coefficients %in% "(Intercept)"]
# training_data_reduced <- training_data[, c("income", selected_vars)]
# testing_data_reduced <- testing_data[, c("income", selected_vars)]
# 
# lasso_reduced <- train(
#   income ~ .,
#   data = training_data_reduced,
#   method = "glmnet",
#   trControl = train_control,
#   tuneGrid = expand.grid(alpha = 1, lambda = lambda_grid)
# )
# 
# ridge_reduced <- train(
#   income ~ .,
#   data = training_data_reduced,
#   method = "glmnet",
#   trControl = train_control,
#   tuneGrid = expand.grid(alpha = 0, lambda = lambda_grid)
# )
# 
# cat("Lasso accuracy:", max(lasso_reduced$results$Accuracy), "\n")
# cat("Ridge accuracy:", max(ridge_reduced$results$Accuracy), "\n")


#################
# Question (vi)
#################

##############
# Part (a)
##############

# Bagging is a method to improve model accuracy by reducing variance. It works by creating 
# multiple bootstrap samples from the original dataset. For classification problems, it uses majority
# voting to decide the final prediction, while for regression, it takes the average of all predictions.

# Random forest improves bagging by adding more randomness. It creates different data samples
# for each tree and also uses a random subset of features at each split.

##############
# Part (b)
##############

# library(caret)
# library(randomForest)
# 
# train_control <- trainControl(method = "cv", number = 5)
# 
# 
# tune_grid <- expand.grid(
#   .mtry = c(2, 5, 9) 
# )
# 
# train_rf <- function(ntree) {
#   train(
#     income ~ ., 
#     data = training_data,
#     method = "rf",
#     trControl = train_control,
#     tuneGrid = tune_grid,
#     ntree = ntree
#   )
# }
# 
# rf_100 <- train_rf(ntree = 100)
# rf_200 <- train_rf(ntree = 200)
# rf_300 <- train_rf(ntree = 300)

##############
# Part (c)
##############

# acc_100 <- max(rf_100$results$Accuracy)
# acc_200 <- max(rf_200$results$Accuracy)
# acc_300 <- max(rf_300$results$Accuracy)
# 
# cat("Accuracy for 100 trees:", acc_100, "\n")
# cat("Accuracy for 200 trees:", acc_200, "\n")
# cat("Accuracy for 300 trees:", acc_300, "\n")

##############
# Part (d)
##############

# cat("Best accuracy from random forest:", max(acc_100, acc_200, acc_300), "\n")
# cat("Best accuracy from lasso/ridge regression:", max(lasso_reduced$results$Accuracy), "\n")


##############
# Part (e)
##############

# best_rf <- rf_300
# predictions <- predict(best_rf, newdata = training_data)
# 
# conf_matrix <- confusionMatrix(predictions, training_data$income)
# 
# print(conf_matrix)
# 
# false_positives <- conf_matrix$table[2, 1] 
# false_negatives <- conf_matrix$table[1, 2]
# 
# cat("False Positives:", false_positives, "\n")
# cat("False Negatives:", false_negatives, "\n")



#################
# Question (vii)
#################

# best_model <- rf_300
# test_predictions <- predict(best_model, newdata = testing_data)
# 
# test_accuracy <- mean(test_predictions == testing_data$income)
# cat("Classification accuracy on the testing set:", test_accuracy, "\n")


# The classification accuracy of the best model on the testing set is 81.72%. This means 
# the model correctly predicted whether income was above or below $50K for about 82% of the individuals in the testing data.




