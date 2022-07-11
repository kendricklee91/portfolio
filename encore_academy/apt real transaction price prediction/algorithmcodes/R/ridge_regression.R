install.packages("glmnet")
install.packages("ISLR")
install.packages("dplyr")
install.packages("tidyr")

library(ISLR)
library(glmnet)
library(dplyr)
library(tidyr)

rawdata <- read.csv("./data/realestateDataForAnalysis5_Ridge_Lasso.csv")
rawdata2 = na.omit(rawdata)

x = model.matrix(real_price~., rawdata2)[,-1]
y = rawdata2 %>% select(real_price) %>% unlist() %>% as.numeric()

grid = 10^seq(10, -2, length = 100)
ridge_mod = glmnet(x, y, alpha = 0, lambda = grid)
dim(coef(ridge_mod))
plot(ridge_mod)

ridge_mod$lambda[50]
coef(ridge_mod)[,50]
sqrt(sum(coef(ridge_mod)[-1,50]^2))

ridge_mod$lambda[60]
coef(ridge_mod)[,60]
sqrt(sum(coef(ridge_mod)[-1,60]^2))

predict(ridge_mod, s = 50, type = "coefficients")[1:20,]

set.seed(1)

train = rawdata2 %>%
  sample_frac(0.8)

test = rawdata2 %>%
  setdiff(train)

x_train = model.matrix(real_price~., train)[,-1]
x_test = model.matrix(real_price~., test)[,-1]

y_train = train %>% select(real_price) %>% unlist() %>% as.numeric()

y_test = test %>% select(real_price) %>% unlist() %>% as.numeric()

ridge_mod = glmnet(x_train, y_train, alpha=0, lambda = grid, thresh = 1e-12)
ridge_pred = predict(ridge_mod, s = 4, newx = x_test)
mean((ridge_pred - y_test)^2)

mean((mean(y_train) - y_test)^2)

ridge_pred = predict(ridge_mod, s = 1e10, newx = x_test)
mean((ridge_pred - y_test)^2)

ridge_pred = predict(ridge_mod, s = 0, newx = x_test, exact = T)
mean((ridge_pred - y_test)^2)

lm(real_price~., data = train)
predict(ridge_mod, s = 0, exact = T, type="coefficients")[1:20,]

set.seed(1)
cv.out = cv.glmnet(x_train, y_train, alpha = 0) # Fit ridge regression model on training data
bestlam = cv.out$lambda.min  # Select lamda that minimizes training MSE
bestlam

plot(cv.out) # Draw plot of training MSE as a function of lambda

ridge_pred = predict(ridge_mod, s = bestlam, newx = x_test) # Use best lambda to predict test data
mean((ridge_pred - y_test)^2) # Calculate test MSE

out = glmnet(x, y, alpha = 0) # Fit ridge regression model on full dataset
predict(out, type = "coefficients", s = bestlam)[1:20,] # Display coefficients using lambda chosen by CV


