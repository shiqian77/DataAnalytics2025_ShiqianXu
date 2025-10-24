##########################################
### Principal Component Analysis (PCA) ###
##########################################

library(caret)
library(ggfortify)

# PCA with iris dataset
iris.df <- iris
head(iris.df)

# creating another dataframe from iris dataset that contains the columns from 1 to 4
X <- iris.df[,1:4]
Y <- iris.df[,5]

## scatter plot of 2 variables colored by class
ggplot(X, aes(x = Petal.Length, y = Petal.Width, color = Y, fill = Y)) + geom_point() + 
  stat_ellipse(type = "t",geom = "polygon",alpha = 0.4)

## feature-class plots
featurePlot(x=X, y=Y, plot="ellipse")

featurePlot(x=X, y=Y, plot="box")

scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=X, y=Y, plot="density", scales=scales)

## Variance/Covariance/correlation

var(X$Petal.Width)
var(X$Petal.Length)
cov(X$Petal.Length,X$Petal.Width)
cor(X$Petal.Length,X$Petal.Width)

## scatter plot of 2 variables
ggplot(X, aes(x = Petal.Length, y = Petal.Width)) + geom_point(color="blue")

var(X$Sepal.Width)
var(X$Sepal.Length)
cov(X$Sepal.Length,X$Sepal.Width)
cor(X$Sepal.Length,X$Sepal.Width)

## scatter plot of 2 variables
ggplot(X, aes(x = Sepal.Length, y = Sepal.Width)) + geom_point(color="blue")


####### PCA #######

Xmat <- as.matrix(X)

Xc <- scale(Xmat, center = T, scale = F)

principal_components <- princomp(Xc)

summary(principal_components)

principal_components$loadings


# using the plot() function, we can plot the principal components.
plot(principal_components)

# plotting the principal_components using the a line in plot() functions 
plot(principal_components, type = "l")

# using rhw biplot() function we can plot the components
biplot(principal_components)

## using autoplot() function to plot the components
autoplot(principal_components, data = iris, colour = 'Species',
         loadings = TRUE, loadings.colour = 'blue',
         loadings.label = TRUE, loadings.label.size = 3, scale = 0)

## scatter plot of dataset in PCs
ggplot(principal_components$scores, aes(x = Comp.1, y = Comp.2)) + geom_point(color="blue")

# loadings
principal_components$loadings


####### PCA Manually #########

C <- cov(Xc)

C.eigens <-  eigen(C)

C.eigens$vectors
C.eigens$values

V <- C.eigens$vectors

T <- X %*% V

## scatter plot of dataset in PCs
ggplot(T, aes(x = T[,1], y = -T[,2])) + geom_point(color="blue")


####### Exercise #######

library(MASS)

Boston.c <- Boston

principal_components <- princomp(Boston.c)

principal_components$loadings

summary(principal_components)


####### THE END #######
