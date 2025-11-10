setwd("~/Desktop/itws_6600/lab5")

suppressPackageStartupMessages({
  library(e1071)        
  library(caret)        
  library(dplyr)
  library(randomForest)
})

wine.data <- read.csv("wine.data", header = FALSE)

## Assign column names
names(wine.data) <- c(
  "Class","Alcohol","Malic.acid","Ash","Alcalinity.of.ash","Magnesium",
  "Total.phenols","Flavanoids","Nonflavanoid.phenols","Proanthocyanins",
  "Color.int","Hue","OD280.OD315","Proline"
)
wine.data$Class <- factor(wine.data$Class)

## Choose subset of features 
feats <- c("Alcohol","Flavanoids","Color.int","Proline")
wine.data[, feats] <- scale(wine.data[, feats])   # scale only chosen features

## Stratified Train/Test split (70/30) 
set.seed(42)
train.idx <- caret::createDataPartition(wine.data$Class, p = 0.70, list = FALSE)
train <- wine.data[train.idx, c("Class", feats)]
test  <- wine.data[-train.idx, c("Class", feats)]

## Helper functions 
cm.metrics <- function(cm) {
  diagv <- diag(cm)
  prec  <- diagv / colSums(cm)
  rec   <- diagv / rowSums(cm)
  f1    <- 2 * prec * rec / (prec + rec)
  res <- data.frame(
    class = rownames(cm),
    precision = round(prec, 4),
    recall    = round(rec, 4),
    f1        = round(f1, 4),
    row.names = NULL
  )
  rbind(res,
        data.frame(
          class     = "macro_avg",
          precision = round(mean(prec, na.rm = TRUE), 4),
          recall    = round(mean(rec,  na.rm = TRUE), 4),
          f1        = round(mean(f1,   na.rm = TRUE), 4)
        ))
}

evaluate.model <- function(model, test) {
  pred <- predict(model, newdata = test)
  cm   <- table(Actual = test$Class, Predicted = pred)
  list(cm = cm, metrics = cm.metrics(cm))
}

## SVM #1 — Linear kernel (tune C)
set.seed(42)
tune.linear <- tune.svm(
  Class ~ ., data = train,
  kernel = "linear",
  cost   = 2^(-6:8)
)
svm.linear <- tune.linear$best.model
cat("\n[Linear SVM] Best C =", svm.linear$cost, "\n")

## SVM #2 — RBF kernel (tune C & gamma)
set.seed(42)
tune.rbf <- tune.svm(
  Class ~ ., data = train,
  kernel = "radial",
  cost   = 2^(-6:8),
  gamma  = 2^(-10:4)
)
svm.rbf <- tune.rbf$best.model
cat("\n[RBF SVM] Best C =", svm.rbf$cost, " | Best gamma =", svm.rbf$gamma, "\n")

## Random Forest (using same features)
p <- length(feats)
set.seed(42)
rf.model <- randomForest(
  Class ~ ., data = train,
  ntree = 500,
  mtry  = max(1, floor(sqrt(p))),   
  importance = TRUE
)

## Evaluation
res.lin.tr <- evaluate.model(svm.linear, train)
res.rbf.tr <- evaluate.model(svm.rbf,    train)
res.rf.tr  <- evaluate.model(rf.model,   train)

res.lin.te <- evaluate.model(svm.linear, test)
res.rbf.te <- evaluate.model(svm.rbf,    test)
res.rf.te  <- evaluate.model(rf.model,   test)

## Display results
cat("\n================ TRAIN ================\n")
cat("\nLinear SVM Confusion Matrix:\n"); print(res.lin.tr$cm); print(res.lin.tr$metrics)
cat("\nRBF SVM Confusion Matrix:\n");    print(res.rbf.tr$cm); print(res.rbf.tr$metrics)
cat("\nRandom Forest Confusion Matrix:\n"); print(res.rf.tr$cm);  print(res.rf.tr$metrics)

cat("\n================ TEST =================\n")
cat("\nLinear SVM Confusion Matrix:\n"); print(res.lin.te$cm); print(res.lin.te$metrics)
cat("\nRBF SVM Confusion Matrix:\n");    print(res.rbf.te$cm); print(res.rbf.te$metrics)
cat("\nRandom Forest Confusion Matrix:\n"); print(res.rf.te$cm);  print(res.rf.te$metrics)

## Test summary 
pick.macro <- function(res) subset(res$metrics, class == "macro_avg")[, c("precision", "recall", "f1")]
cmp <- rbind(
  `SVM-linear`   = pick.macro(res.lin.te),
  `SVM-RBF`      = pick.macro(res.rbf.te),
  `RandomForest` = pick.macro(res.rf.te)
)
cat("\n==== Test Macro-Average (Precision / Recall / F1) ====\n")
print(cmp)
