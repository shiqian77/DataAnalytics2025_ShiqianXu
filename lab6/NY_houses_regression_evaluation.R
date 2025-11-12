################################
# Evaluating Regression Models #
################################

library(readr)
library(ggplot2)
library(e1071)
library(caret)
library(cv)

## read data
NY_House_Dataset <- read_csv("~/Courses/Data Analytics/Fall25/datasets/NY-House-Dataset.csv")

dataset <- NY_House_Dataset

## column names
names(dataset)

## Plot dataset
ggplot(dataset, aes(x = PROPERTYSQFT, y = PRICE)) +
  geom_point()

ggplot(dataset, aes(x = log10(PROPERTYSQFT), y = log10(PRICE))) +
  geom_point()


## Linear Regression Model
lin.mod0 <- lm(log10(PRICE) ~ log10(PROPERTYSQFT), dataset)

summary(lin.mod0)

ggplot(dataset, aes(x = log10(PROPERTYSQFT), y = log10(PRICE))) +
  geom_point() +
  stat_smooth(method = "lm", col="blue")

ggplot(lin.mod0, aes(x = .fitted, y = .resid)) +
  geom_point() +
  geom_hline(yintercept = 0)



## Cleaning data
dataset.sub0 <- dataset[-which(dataset$PROPERTYSQFT==2184.207862 | dataset$PRICE>20000000),]


## linear model with clean subset
lin.mod1 <- lm(log10(PRICE) ~ log10(PROPERTYSQFT), dataset.sub0)
summary(lin.mod1)

ggplot(dataset.sub0, aes(x = log10(PROPERTYSQFT), y = log10(PRICE))) +
  geom_point() +
  stat_smooth(method = "lm", col="blue")

ggplot(lin.mod1, aes(x = .fitted, y = .resid)) +
  geom_point() +
  geom_hline(yintercept = 0)


## SVM - linear
svm.mod0 <- svm(log10(PRICE) ~ log10(PROPERTYSQFT), dataset.sub0, kernel="linear")

summary(svm.mod0)

svm.pred0 <- data.frame(real = log10(dataset.sub0$PRICE),predicted=predict(svm.mod0, dataset.sub0))

ggplot(dataset.sub0, aes(x = log10(PROPERTYSQFT), y = log10(PRICE))) +
  geom_point() +
  geom_line(aes(x=log10(PROPERTYSQFT), y=svm.pred0$predicted), col="green")


ggplot(svm.pred0, aes(x = predicted, y = real-predicted)) +
  geom_point() +
  geom_hline(yintercept = 0)


## SVM - radial
svm.mod1 <- svm(log10(PRICE) ~ log10(PROPERTYSQFT), dataset.sub0, kernel="radial")

# summary(svm.mod1)

svm.pred1 <- data.frame(real = log10(dataset.sub0$PRICE),predicted=predict(svm.mod1, dataset.sub0))

ggplot(dataset.sub0, aes(x = log10(PROPERTYSQFT), y = log10(PRICE))) +
  geom_point() +
  geom_line(aes(x=log10(PROPERTYSQFT), y=svm.pred1$predicted), col="red")

ggplot(svm.pred1, aes(x = predicted, y = real-predicted)) +
  geom_point() +
  geom_hline(yintercept = 0)

## SVM - radial - optimized
## set ranges for parameters to tune
# gamma.range <- seq(0.1,10, .1)
gamma.range <- 10^seq(-3,2,1)
gamma.range

# C.range <- seq(1,20, 1)
C.range <- 10^seq(-3,2,1)
C.range

tuned.svm <- tune.svm(log10(PRICE) ~ log10(PROPERTYSQFT), data=dataset.sub0, kernel="radial", gamma = gamma.range, cost = C.range, tune.control=tune.control(cross = 5))
tuned.svm

opt.gamma <- tuned.svm$best.parameters$gamma
opt.C <- tuned.svm$best.parameters$cost

svm.mod2 <- svm(log10(PRICE) ~ log10(PROPERTYSQFT), dataset.sub0, kernel="radial", gamma=opt.gamma, cost=opt.C)

# summary(svm.mod2)

svm.pred2 <- data.frame(real = log10(dataset.sub0$PRICE),predicted=predict(svm.mod2, dataset.sub0))

ggplot(dataset.sub0, aes(x = log10(PROPERTYSQFT), y = log10(PRICE))) +
  geom_point() +
  geom_line(aes(x=log10(PROPERTYSQFT), y=svm.pred2$predicted), col="red")

ggplot(svm.pred2, aes(x = predicted, y = real-predicted)) +
  geom_point() +
  geom_hline(yintercept = 0)

## Estimating Model Errors

## split train/test
train.indexes <- sample(nrow(dataset.sub0),0.75*nrow(dataset.sub0))

train <- dataset.sub0[train.indexes,]
test <- dataset.sub0[-train.indexes,]

## LM
lin.mod0 <- lm(log10(PRICE) ~ log10(PROPERTYSQFT), train)

## Use CV function to run 
cv.res <- cv(lin.mod0)
  
summary(cv.res)

lm.pred0 <- predict(lin.mod0, test)

## err = predicted - real
err <- lm.pred0-log10(test$PRICE)

## MAE
abs.err <- abs(err)
mean.abs.err <- mean(abs.err)
mean.abs.err

## MSE
sq.err <- err^2
mean.sq.err <- mean(sq.err)
mean.sq.err

## RMSE
sq.err <- err^2
mean.sq.err <- mean(sq.err)
root.mean.sq.err <- sqrt(mean.sq.err)
root.mean.sq.err

### Cross Validation ###

### Monte Carlo CV

## Lin Model
k = 100
mae0 <- c()
mse0 <- c()
rmse0 <- c()

for (i in 1:k) {
  
  train.indexes <- sample(nrow(dataset.sub0),0.75*nrow(dataset.sub0))
  
  train <- dataset.sub0[train.indexes,]
  test <- dataset.sub0[-train.indexes,]
  
  lin.mod <- lm(log10(PRICE) ~ log10(PROPERTYSQFT), train)
  
  lm.pred <- predict(lin.mod, test)  
  
  err <- lm.pred-log10(test$PRICE)
  
  abs.err <- abs(err)
  mean.abs.err <- mean(abs.err)
  
  sq.err <- err^2
  mean.sq.err <- mean(sq.err)
  
  root.mean.sq.err <- sqrt(mean.sq.err)  
  
  mae0 <- c(mae0,mean.abs.err)
  mse0 <- c(mse0,mean.sq.err)
  rmse0 <- c(rmse0,root.mean.sq.err)
}


results0 <- data.frame(mae=mean(mae0), mse=mean(mse0), rmse=mean(rmse0))
results0

## Linear SVM Model
k = 100
mae1 <- c()
mse1 <- c()
rmse1 <- c()

for (i in 1:k) {
  train.indexes <- sample(nrow(dataset.sub0),0.75*nrow(dataset.sub0))
  
  train <- dataset.sub0[train.indexes,]
  test <- dataset.sub0[-train.indexes,]
  
  svm.mod <- svm(log10(PRICE) ~ log10(PROPERTYSQFT), dataset.sub0, kernel="radial")
  
  svm.pred <- predict(svm.mod, test)  
  
  err <- svm.pred-log10(test$PRICE)
  
  abs.err <- abs(err)
  mean.abs.err <- mean(abs.err)
  
  sq.err <- err^2
  mean.sq.err <- mean(sq.err)
  
  root.mean.sq.err <- sqrt(mean.sq.err)  
  
  mae1 <- c(mae1,mean.abs.err)
  mse1 <- c(mse1,mean.sq.err)
  rmse1 <- c(rmse1,root.mean.sq.err)
}

mean(mae1)
mean(mse1)
mean(rmse1)

results1 <- data.frame(mae=mean(mae1), mse=mean(mse1), rmse=mean(rmse1))
results1

## Radial SVM Model
k = 100
mae2 <- c()
mse2 <- c()
rmse2 <- c()

for (i in 1:k) {
  train.indexes <- sample(nrow(dataset.sub0),0.75*nrow(dataset.sub0))
  
  train <- dataset.sub0[train.indexes,]
  test <- dataset.sub0[-train.indexes,]
  
  svm.mod <- svm(log10(PRICE) ~ log10(PROPERTYSQFT), dataset.sub0, kernel="radial", gamma=opt.gamma, cost=opt.C)
  
  svm.pred <- predict(svm.mod, test)  
  
  err <- svm.pred-log10(test$PRICE)
  
  abs.err <- abs(err)
  mean.abs.err <- mean(abs.err)
  
  sq.err <- err^2
  mean.sq.err <- mean(sq.err)
  root.mean.sq.err <- sqrt(mean.sq.err)  
  
  mae2 <- c(mae2,mean.abs.err)
  mse2 <- c(mse2,mean.sq.err)
  rmse2 <- c(rmse2,root.mean.sq.err)
}

mean(mae2)
mean(mse2)
mean(rmse2)

results2 <- data.frame(mae=mean(mae2), mse=mean(mse2), rmse=mean(rmse2))
results2


results <- rbind(results0,results1,results2)
results

#### THE END ####