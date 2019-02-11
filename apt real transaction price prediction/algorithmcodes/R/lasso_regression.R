install.packages("glmnet")
install.packages("ISLR")
install.packages("dplyr")
install.packages("tidyr")

library(ISLR)
library(glmnet)
library(dplyr)
library(tidyr)

rawdata <- read.csv("../project_sdh/realestateDataForAnalysis5_Ridge3.csv")

summary(rawdata)

tail(rawdata$heat_fuel_category_cogeneration)

rawdata2 = na.omit(rawdata)

x = model.matrix(real_price~., rawdata2)[,-1]
y = rawdata2 %>% select(real_price) %>% unlist() %>% as.numeric()

grid = 10^seq(10, -2, length = 100)
lasso_mod = glmnet(x, y, alpha = 1, lambda = grid)
dim(coef(lasso_mod))
plot(lasso_mod)

lasso_mod$lambda[50]
coef(lasso_mod)[,50]
sqrt(sum(coef(lasso_mod)[-1,50]^2))

lasso_mod$lambda[60]
coef(lasso_mod)[,60]
sqrt(sum(coef(lasso_mod)[-1,60]^2))

predict(lasso_mod, s = 50, type = "coefficients")[1:20,]

set.seed(1)

train = rawdata2 %>% sample_frac(0.8)

test = rawdata2 %>% setdiff(train)

x_train = model.matrix(real_price~., train)[,-1]
x_test = model.matrix(real_price~., test)[,-1]

y_train = train %>% select(real_price) %>% unlist() %>% as.numeric()

y_test = test %>% select(real_price) %>% unlist() %>% as.numeric()

lasso_mod = glmnet(x_train, y_train, alpha = 1, lambda = grid) # Fit lasso model on training data
plot(lasso_mod)    # Draw plot of coefficients

set.seed(1)
cv.out = cv.glmnet(x_train, y_train, alpha = 1) # Fit lasso model on training data
plot(cv.out) # Draw plot of training MSE as a function of lambda
bestlam = cv.out$lambda.min # Select lamda that minimizes training MSE
lasso_pred = predict(lasso_mod, s = bestlam, newx = x_test) # Use best lambda to predict test data
mean((lasso_pred - y_test)^2) # Calculate test MSE

out = glmnet(x, y, alpha = 1, lambda = grid) # Fit lasso model on full dataset
lasso_coef = predict(out, type = "coefficients", s = bestlam)[1:20,] # Display coefficients using lambda chosen by CV
lasso_coef

lasso_coef[lasso_coef != 0] # Display only non-zero coefficients

sqrt(mean((lasso_pred - y_test)^2))