###############################
### Support Vector Machines ###
###############################


library("caret")
library(e1071)

## take copy
dataset <- iris

dataset$Species <- as.character(dataset$Species)
dataset <- dataset[-which(dataset$Species=="setosa"),]
dataset$Species <- as.factor(dataset$Species)

# ## split train/test
n <- nrow(dataset)
train.indexes <- sample(n,0.7*n)

train <- dataset[train.indexes,]
test <- dataset[-train.indexes,]

## separate x (features) & y (class labels)
X <- train[,1:4] 
Y <- train[,5]

## feature boxplots
boxplot(X, main="iris features")

## class label distributions
plot(Y)

## feature-class plots
featurePlot(x=X, y=Y, plot="ellipse")

featurePlot(x=X, y=Y, plot="box")

scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=X, y=Y, plot="density", scales=scales)

ggplot(train, aes(x = Petal.Length, y = Petal.Width, colour = Species)) +
  geom_point()


## train SVM model - linear kernel
svm.mod0 <- svm(Species ~ Petal.Length + Petal.Width, data = train, kernel = 'linear')

svm.mod0

plot(svm.mod0, data = train, formula = Petal.Length~Petal.Width, svSymbol = "x", dataSymbol = "o")


train.pred <- predict(svm.mod0, train)

cm = as.matrix(table(Actual = train$Species, Predicted = train.pred))

cm

n = sum(cm) # number of instances
nc = nrow(cm) # number of classes
diag = diag(cm) # number of correctly classified instances per class 
rowsums = apply(cm, 1, sum) # number of instances per class
colsums = apply(cm, 2, sum) # number of predictions per class
p = rowsums / n # distribution of instances over the actual classes
q = colsums / n # distribution of instances over the predicted 

recall = diag / rowsums 
precision = diag / colsums
f1 = 2 * precision * recall / (precision + recall) 

data.frame(precision, recall, f1)

make.grid = function(x, n = 75) {
  grange = apply(x, 2, range)
  x1 = seq(from = grange[1,1], to = grange[2,1], length = n)
  x2 = seq(from = grange[1,2], to = grange[2,2], length = n)
  expand.grid(Petal.Length = x1, Petal.Width = x2)
}

x <- train[,3:4]
y <- as.numeric(train$Species)
y[y==2] <- -1

xgrid = make.grid(x)
xgrid[1:10,]

ygrid = predict(svm.mod0, xgrid)

plot(xgrid, col = c("red","blue")[as.numeric(ygrid)], pch = 20, cex = .2)

points(x, col = y + 3, pch = 19)
points(x[svm.mod0$index,], pch = 5, cex = 2)

# beta = drop(t(svm.mod0$coefs)%*%as.matrix(x)[svm.mod0$index,])
# beta0 = svm.mod0$rho
# 
# plot(xgrid, col = c("red", "blue")[as.numeric(ygrid)], pch = 20, cex = .2)
# points(x, col = y + 3, pch = 19)
# points(x[svm.mod0$index,], pch = 5, cex = 2)
# abline(beta0 / beta[2], -beta[1] / beta[2])
# abline((beta0 - 1) / beta[2], -beta[1] / beta[2], lty = 2)
# abline((beta0 + 1) / beta[2], -beta[1] / beta[2], lty = 2)

## train SVM model - polynomial kernel
svm.mod1 <- svm(Species ~ Petal.Length+Petal.Width, data = train, kernel = 'radial')

svm.mod1

plot(svm.mod1, train, Petal.Width~Petal.Length)

train.pred <- predict(svm.mod1, train)

x <- train[,3:4]
y <- as.numeric(train$Species)
y[y==2] <- -1

xgrid = make.grid(x)
xgrid[1:10,]

ygrid = predict(svm.mod1, xgrid)

plot(xgrid, col = c("red","blue")[as.numeric(ygrid)], pch = 20, cex = .2)

points(x, col = y + 3, pch = 19)
points(x[svm.mod1$index,], pch = 5, cex = 2)

cm = as.matrix(table(Actual = train$Species, Predicted = train.pred))

cm

n = sum(cm) # number of instances
nc = nrow(cm) # number of classes
diag = diag(cm) # number of correctly classified instances per class 
rowsums = apply(cm, 1, sum) # number of instances per class
colsums = apply(cm, 2, sum) # number of predictions per class
p = rowsums / n # distribution of instances over the actual classes
q = colsums / n # distribution of instances over the predicted 

recall = diag / rowsums 
precision = diag / colsums
f1 = 2 * precision * recall / (precision + recall) 

data.frame(precision, recall, f1)


## Tuned SVM - polynomial
tuned.svm <- tune.svm(Species ~ Petal.Length + Petal.Width, data = train, kernel = 'polynomial',gamma = seq(1/2^nrow(iris),1, .01), cost = 2^seq(-6, 4, 2))

tuned.svm

svm.mod2 <- svm(Species ~ Petal.Length + Petal.Width, data = train, kernel = 'polynomial', gamma = .81, cost = 1)

svm.mod2

train.pred <- predict(svm.mod2, train)

x <- train[,3:4]
y <- as.numeric(train$Species)
y[y==2] <- -1

xgrid = make.grid(x)
# xgrid[1:10,]

ygrid = predict(svm.mod2, xgrid)

plot(xgrid, col = c("red","blue")[as.numeric(ygrid)], pch = 20, cex = .2)

points(x, col = y + 3, pch = 19)
points(x[svm.mod2$index,], pch = 5, cex = 2)

cm = as.matrix(table(Actual = train$Species, Predicted = train.pred))

cm

n = sum(cm) # number of instances
nc = nrow(cm) # number of classes
diag = diag(cm) # number of correctly classified instances per class 
rowsums = apply(cm, 1, sum) # number of instances per class
colsums = apply(cm, 2, sum) # number of predictions per class
p = rowsums / n # distribution of instances over the actual classes
q = colsums / n # distribution of instances over the predicted 

recall = diag / rowsums 
precision = diag / colsums
f1 = 2 * precision * recall / (precision + recall) 

data.frame(precision, recall, f1)


### Test set prediction ###

## model 0 - linear kernel
test.pred <- predict(svm.mod0, test)

cm = as.matrix(table(Actual = test$Species, Predicted = test.pred))

cm

n = sum(cm) # number of instances
nc = nrow(cm) # number of classes
diag = diag(cm) # number of correctly classified instances per class 
rowsums = apply(cm, 1, sum) # number of instances per class
colsums = apply(cm, 2, sum) # number of predictions per class
p = rowsums / n # distribution of instances over the actual classes
q = colsums / n # distribution of instances over the predicted 

recall = diag / rowsums 
precision = diag / colsums
f1 = 2 * precision * recall / (precision + recall) 

data.frame(precision, recall, f1)

## model 1 - polynomial kernel
test.pred <- predict(svm.mod1, test)

cm = as.matrix(table(Actual = test$Species, Predicted = test.pred))

cm

n = sum(cm) # number of instances
nc = nrow(cm) # number of classes
diag = diag(cm) # number of correctly classified instances per class 
rowsums = apply(cm, 1, sum) # number of instances per class
colsums = apply(cm, 2, sum) # number of predictions per class
p = rowsums / n # distribution of instances over the actual classes
q = colsums / n # distribution of instances over the predicted 

recall = diag / rowsums 
precision = diag / colsums
f1 = 2 * precision * recall / (precision + recall) 

data.frame(precision, recall, f1)

## model 2- polynomial kernel with tuned parameters
test.pred <- predict(svm.mod2, test)

cm = as.matrix(table(Actual = test$Species, Predicted = test.pred))

cm

n = sum(cm) # number of instances
nc = nrow(cm) # number of classes
diag = diag(cm) # number of correctly classified instances per class 
rowsums = apply(cm, 1, sum) # number of instances per class
colsums = apply(cm, 2, sum) # number of predictions per class
p = rowsums / n # distribution of instances over the actual classes
q = colsums / n # distribution of instances over the predicted 

recall = diag / rowsums 
precision = diag / colsums
f1 = 2 * precision * recall / (precision + recall) 

data.frame(precision, recall, f1)


##########################################

