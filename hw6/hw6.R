suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(ggplot2)
  library(tidyr)
  library(class)
  library(randomForest)
  library(e1071)
  library(cluster)
})

setwd("~/Desktop/itws_6600/hw6")

names.file <- "communities.names"
data.file  <- "communities.data"

stopifnot(file.exists(names.file), file.exists(data.file))

nm.lines  <- readLines(names.file)
attr.lines <- nm.lines[grepl("^@attribute", nm.lines, ignore.case = TRUE)]

## extract second token as attribute name
get_attr_name <- function(line) {
  parts <- strsplit(line, "[[:space:]]+")[[1]]
  parts <- parts[parts != ""]
  if (length(parts) >= 2) parts[2] else NA_character_
}
col.names <- sapply(attr.lines, get_attr_name)
col.names <- unname(col.names)

comm <- read.table(data.file,
                   sep = ",",
                   header = FALSE,
                   na.strings = "?",
                   stringsAsFactors = FALSE)
stopifnot(ncol(comm) == length(col.names))
names(comm) <- col.names

cat("Rows x Cols:", nrow(comm), "x", ncol(comm), "\n")
cat("First few column names:\n")
print(head(names(comm), 15))

## goal variable
target <- "ViolentCrimesPerPop"
stopifnot(target %in% names(comm))

id.cols <- c("state", "county", "community", "communityname", "fold")
id.cols <- intersect(id.cols, names(comm))

## drop IDs + target
is.num <- sapply(comm, is.numeric)
feature.cols <- setdiff(names(comm)[is.num], c(id.cols, target))

## remove rows with missing target
comm <- comm[!is.na(comm[[target]]), ]
cat("After dropping rows with NA target:", nrow(comm), "rows\n")

## ---------- Output Plots ----------
out.dir <- "plots_a6"
if (!dir.exists(out.dir)) dir.create(out.dir)

## 1. Exploratory Data Analysis (EDA)       
cat("\n================= EDA =================\n")

## basic summary of target
cat("\n--- Summary of", target, "---\n")
print(summary(comm[[target]]))

## histogram of target + density
p.tgt.hist <- ggplot(comm, aes(x = .data[[target]])) +
  geom_histogram(bins = 40, fill = "skyblue", alpha = 0.7) +
  labs(title = paste("Histogram of", target),
       x = target, y = "Count")
ggsave(file.path(out.dir, "01_hist_ViolentCrimesPerPop.png"),
       p.tgt.hist, width = 7, height = 5, dpi = 160)

p.tgt.dens <- ggplot(comm, aes(x = .data[[target]])) +
  geom_density(fill = "steelblue", alpha = 0.5) +
  labs(title = paste("Density of", target),
       x = target, y = "Density")
ggsave(file.path(out.dir, "02_density_ViolentCrimesPerPop.png"),
       p.tgt.dens, width = 7, height = 5, dpi = 160)

## boxplot for target (outlier check)
p.tgt.box <- ggplot(comm, aes(y = .data[[target]])) +
  geom_boxplot(outlier.alpha = 0.4) +
  labs(title = paste("Boxplot of", target),
       x = "", y = target)
ggsave(file.path(out.dir, "03_boxplot_ViolentCrimesPerPop.png"),
       p.tgt.box, width = 5, height = 5, dpi = 160)

eda.vars <- c(
  "medIncome",
  "PctPopUnderPov",
  "PctUnemployed",
  "PctBSorMore",
  "racepctblack",
  "racePctWhite",
  "racePctHisp",
  "pctUrban",
  "PopDens"
)
eda.vars <- intersect(eda.vars, feature.cols)

cat("\nEDA variables used:\n")
print(eda.vars)

eda.df <- comm[, c(target, eda.vars)]
cat("\n--- Summaries of selected predictors ---\n")
print(summary(eda.df))

## pairwise scatter of target vs each eda variable
for (v in eda.vars) {
  p <- ggplot(eda.df, aes_string(x = v, y = target)) +
    geom_point(alpha = 0.4) +
    geom_smooth(method = "lm", se = FALSE, color = "red") +
    labs(title = paste(target, "vs", v),
         x = v, y = target)
  fn <- paste0("EDA_scatter_", target, "_vs_", v, ".png")
  ggsave(file.path(out.dir, fn), p, width = 7, height = 5, dpi = 160)
}

## correlation of all numeric features with ViolentCrimesPerPop
num.cols <- intersect(feature.cols, names(comm)[is.num])
y <- comm[[target]]

cor.vec <- sapply(num.cols, function(nm) {
  cor(comm[[nm]], y, use = "complete.obs")
})

cor.rank <- sort(cor.vec, decreasing = TRUE)
cor.rank.abs <- sort(abs(cor.vec), decreasing = TRUE)

cat("\nTop 10 features by *positive* correlation with", target, ":\n")
print(head(cor.rank, 10))

cat("\nTop 10 features by |correlation| with", target, ":\n")
print(head(cor.rank.abs, 10))

## pick top-k predictors for modeling
top.k <- 12
top.feats <- names(cor.rank.abs)[1:min(top.k, length(cor.rank.abs))]
cat("\nTop predictors by |cor| for modeling:\n")
print(top.feats)

## scatterplots for 3 strongest predictors
strong3 <- head(top.feats, 3)
for (v in strong3) {
  p <- ggplot(comm, aes(x = .data[[v]], y = .data[[target]])) +
    geom_point(alpha = 0.4) +
    geom_smooth(method = "lm", se = FALSE, color = "darkgreen") +
    labs(title = paste(target, "vs", v, "(strong predictor)"),
         x = v, y = target)
  fn <- paste0("EDA_strong_", target, "_vs_", v, ".png")
  ggsave(file.path(out.dir, fn), p, width = 7, height = 5, dpi = 160)
}

cat("\nEDA plots saved under:", normalizePath(out.dir), "\n")

## 2. Model Development, Validation, Optimization   
cat("\n=========== Model Development ===========\n")

## helpers 
rmse <- function(e) sqrt(mean(e^2, na.rm = TRUE))
mae  <- function(e) mean(abs(e), na.rm = TRUE)

acc_from_tab <- function(tab) {
  if (sum(tab) == 0) return(NA_real_)
  sum(diag(tab)) / sum(tab)
}
macro_precision <- function(tab){
  if (sum(tab) == 0) return(NA_real_)
  prec <- diag(tab) / rowSums(tab)
  mean(prec, na.rm = TRUE)
}
macro_recall <- function(tab){
  if (sum(tab) == 0) return(NA_real_)
  rec <- diag(tab) / colSums(tab)
  mean(rec, na.rm = TRUE)
}
macro_f1 <- function(tab){
  if (sum(tab) == 0) return(NA_real_)
  prec <- diag(tab) / rowSums(tab)
  rec  <- diag(tab) / colSums(tab)
  f1   <- 2 * prec * rec / (prec + rec)
  mean(f1, na.rm = TRUE)
}

model.df <- comm[, c(target, top.feats)]
model.df <- drop_na(model.df)

cat("\nModeling dataframe:", nrow(model.df), "rows x", ncol(model.df), "cols\n")

set.seed(2025)
n <- nrow(model.df)
train.idx.reg <- sample(n, floor(0.7 * n))
test.idx.reg  <- setdiff(seq_len(n), train.idx.reg)

train.reg <- model.df[train.idx.reg, ]
test.reg  <- model.df[test.idx.reg, ]

## ---------------- Model 1: Linear Regression ----------------
cat("\n--- Model 1: Linear Regression on", target, "with top features ---\n")

form.lm <- as.formula(paste(target, "~", paste(top.feats, collapse = " + ")))
lm.mod <- lm(form.lm, data = train.reg)
cat("\n[LM] Summary:\n")
print(summary(lm.mod))

pred.lm <- predict(lm.mod, newdata = test.reg)
err.lm  <- pred.lm - test.reg[[target]]

rmse.lm <- rmse(err.lm)
mae.lm  <- mae(err.lm)

cat("Linear Regression RMSE:", round(rmse.lm, 4),
    " MAE:", round(mae.lm, 4), "\n")

## residual plot
lm.res.df <- data.frame(
  fitted = pred.lm,
  resid  = err.lm
)
p.lm.res <- ggplot(lm.res.df, aes(x = fitted, y = resid)) +
  geom_point(alpha = 0.4) +
  geom_smooth(method = "loess", se = FALSE, color = "blue") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Model 1: Linear Regression Residuals vs Fitted",
       x = "Fitted ViolentCrimesPerPop",
       y = "Residual")
ggsave(file.path(out.dir, "M1_LM_residuals.png"),
       p.lm.res, width = 7, height = 5, dpi = 160)


## ---------------- Model 2: Random Forest Regression ----------------
cat("\n--- Model 2: Random Forest Regression on", target, " ---\n")

p <- length(top.feats)
set.seed(2025)
rf.reg <- randomForest(
  form.lm,
  data = train.reg,
  ntree = 500,
  mtry  = max(1, floor(sqrt(p))),
  importance = TRUE
)
print(rf.reg)

pred.rf <- predict(rf.reg, newdata = test.reg)
err.rf  <- pred.rf - test.reg[[target]]

rmse.rf <- rmse(err.rf)
mae.rf  <- mae(err.rf)

cat("Random Forest RMSE:", round(rmse.rf, 4),
    " MAE:", round(mae.rf, 4), "\n")

## variable importance plot
png(file.path(out.dir, "M2_RF_importance.png"), width = 800, height = 600)
varImpPlot(rf.reg, main = "Random Forest Variable Importance")
dev.off()

rf.imp <- importance(rf.reg)
rf.imp.df <- data.frame(
  Variable = rownames(rf.imp),
  rf.imp,
  row.names = NULL
)

imp.col <- colnames(rf.imp.df)[2]        
rf.imp.df <- rf.imp.df[order(-rf.imp.df[[imp.col]]), ]

cat("\nRandom Forest variable importance (sorted by", imp.col, "):\n")
print(head(rf.imp.df, 10))

rf.imp <- importance(rf.reg)
rf.imp.df <- data.frame(
  Variable = rownames(rf.imp),
  rf.imp,
  row.names = NULL
)

## use the first importance column 
imp.col <- colnames(rf.imp.df)[2]        
rf.imp.df <- rf.imp.df[order(-rf.imp.df[[imp.col]]), ]

cat("\nRandom Forest variable importance (sorted by", imp.col, "):\n")
print(head(rf.imp.df, 10))

cat("\n--- Set up Classification Target: High vs Low Crime ---\n")

## define high/low based on median of target
median.y <- median(model.df[[target]], na.rm = TRUE)
model.cls <- model.df
model.cls$CrimeLevel <- factor(
  ifelse(model.cls[[target]] > median.y, "high", "low"),
  levels = c("low", "high")
)

## drop the numeric target for classification predictors
cls.predictors <- top.feats
cls.df <- model.cls[, c(cls.predictors, "CrimeLevel")]
cls.df <- drop_na(cls.df)

cat("Classification dataframe:", nrow(cls.df), "rows\n")

set.seed(2025)
n.cls <- nrow(cls.df)
idx <- sample(n.cls)
train.n <- floor(0.7 * n.cls)
train.idx.cls <- idx[1:train.n]
test.idx.cls  <- idx[(train.n + 1):n.cls]

train.cls <- cls.df[train.idx.cls, ]
test.cls  <- cls.df[test.idx.cls, ]

## scale predictors using train stats
X.train.raw <- as.matrix(train.cls[, cls.predictors, drop = FALSE])
X.test.raw  <- as.matrix(test.cls[, cls.predictors, drop = FALSE])

mu <- colMeans(X.train.raw, na.rm = TRUE)
sdv <- apply(X.train.raw, 2, sd)
sdv[!is.finite(sdv) | sdv == 0] <- 1

scale_mat <- function(M, mu, sdv) {
  sweep(sweep(M, 2, mu, FUN = "-"), 2, sdv, FUN = "/")
}

X.train <- scale_mat(X.train.raw, mu, sdv)
X.test  <- scale_mat(X.test.raw,  mu, sdv)

y.train <- train.cls$CrimeLevel
y.test  <- test.cls$CrimeLevel

## ---------------- Model 3: kNN Classification (raw features) ----------------
cat("\n--- Model 3: kNN Classification (top predictors, scaled) ---\n")

set.seed(2025)
k.grid <- seq(1, 51, by = 2)
acc.grid <- numeric(length(k.grid))

for (i in seq_along(k.grid)) {
  k <- k.grid[i]
  pred.k <- knn(train = X.train, test = X.test, cl = y.train, k = k)
  acc.grid[i] <- mean(pred.k == y.test)
}

best.k <- k.grid[which.max(acc.grid)]
cat("Best k (by accuracy) =", best.k,
    "with accuracy =", round(max(acc.grid), 4), "\n")

pred.knn <- knn(train = X.train, test = X.test, cl = y.train, k = best.k)
tab.knn  <- table(predicted = pred.knn, actual = y.test)

cat("\n[Model 3] Confusion matrix (kNN, raw features):\n")
print(tab.knn)
cat("Accuracy:", round(acc_from_tab(tab.knn), 4),
    " MacroP:", round(macro_precision(tab.knn), 4),
    " MacroR:", round(macro_recall(tab.knn), 4),
    " MacroF1:", round(macro_f1(tab.knn), 4), "\n")

## ---------------- Model 4: PCA + kNN Classification ----------------
cat("\n--- Model 4: PCA + kNN Classification ---\n")

## PCA on train predictors (scaled)
pca.fit <- prcomp(X.train, center = FALSE, scale. = FALSE)  # already scaled
var.exp <- pca.fit$sdev^2 / sum(pca.fit$sdev^2)
cum.exp <- cumsum(var.exp)

cat("\nPCA variance explained (first 10 PCs):\n")
print(round(var.exp[1:10], 4))
cat("Cumulative (first 10 PCs):\n")
print(round(cum.exp[1:10], 4))

## choose number of PCs to explain ~80% variance
k.pc <- which(cum.exp >= 0.80)[1]
if (is.na(k.pc) || k.pc < 2) k.pc <- min(5, ncol(X.train))

cat("Using", k.pc, "PCs for classification.\n")

Z.train <- pca.fit$x[, 1:k.pc, drop = FALSE]
Z.test  <- predict(pca.fit, newdata = X.test)[, 1:k.pc, drop = FALSE]

## tune k again but on PC space
set.seed(2025)
k.grid2 <- seq(1, 51, by = 2)
acc.grid2 <- numeric(length(k.grid2))

for (i in seq_along(k.grid2)) {
  k <- k.grid2[i]
  pred.k2 <- knn(train = Z.train, test = Z.test, cl = y.train, k = k)
  acc.grid2[i] <- mean(pred.k2 == y.test)
}

best.k2 <- k.grid2[which.max(acc.grid2)]
cat("Best k (PCA space) =", best.k2,
    "with accuracy =", round(max(acc.grid2), 4), "\n")

pred.knn.pca <- knn(train = Z.train, test = Z.test, cl = y.train, k = best.k2)
tab.knn.pca  <- table(predicted = pred.knn.pca, actual = y.test)

cat("\n[Model 4] Confusion matrix (kNN on PCs):\n")
print(tab.knn.pca)
cat("Accuracy:", round(acc_from_tab(tab.knn.pca), 4),
    " MacroP:", round(macro_precision(tab.knn.pca), 4),
    " MacroR:", round(macro_recall(tab.knn.pca), 4),
    " MacroF1:", round(macro_f1(tab.knn.pca), 4), "\n")

## simple line plot comparing acc vs k for raw vs PCA
acc.df <- data.frame(
  k = rep(k.grid, 2),
  acc = c(acc.grid, acc.grid2),
  model = rep(c("kNN_raw", "kNN_PCA"), each = length(k.grid))
)

p.acc <- ggplot(acc.df, aes(x = k, y = acc, color = model)) +
  geom_line() +
  geom_point() +
  labs(title = "kNN Accuracy vs k: Raw vs PCA Features",
       x = "k", y = "Accuracy") +
  theme_bw()
ggsave(file.path(out.dir, "M3_M4_kNN_acc_vs_k.png"),
       p.acc, width = 7, height = 5, dpi = 160)

## ---------------- Model 5: K-Means Clustering ----------------
cat("\n--- Model 5: K-Means clustering on top predictors ---\n")

clust.df <- comm[, top.feats]
clust.df <- drop_na(clust.df)

X.clust <- scale(as.matrix(clust.df))
D.clust <- dist(X.clust)

k.list <- 2:10
sil.k <- numeric(length(k.list))

for (i in seq_along(k.list)) {
  k <- k.list[i]
  set.seed(100 + k)
  km <- kmeans(X.clust, centers = k, nstart = 20)
  sil <- silhouette(km$cluster, D.clust)
  sil.k[i] <- mean(sil[, 3])
}

best.k.km <- k.list[which.max(sil.k)]
cat("Best K for k-means (by avg silhouette) =", best.k.km, "\n")

set.seed(100 + best.k.km)
km.best <- kmeans(X.clust, centers = best.k.km, nstart = 50)

## attach cluster labels + inspect mean crime rate per cluster
clust.result <- comm[rownames(clust.df), ]
clust.result$Cluster <- factor(km.best$cluster)

crime.by.cluster <- clust.result %>%
  group_by(Cluster) %>%
  summarise(
    mean_ViolentCrimesPerPop = mean(.data[[target]], na.rm = TRUE),
    n = n()
  )
cat("\nMean violent crime rate per cluster:\n")
print(crime.by.cluster)

p.clust.crime <- ggplot(crime.by.cluster,
                        aes(x = Cluster, y = mean_ViolentCrimesPerPop)) +
  geom_col(fill = "orange", alpha = 0.8) +
  labs(title = "Average ViolentCrimesPerPop by Cluster (K-Means)",
       x = "Cluster", y = "Mean ViolentCrimesPerPop") +
  theme_bw()
ggsave(file.path(out.dir, "M5_kmeans_cluster_crime.png"),
       p.clust.crime, width = 6, height = 5, dpi = 160)

p.sil <- ggplot(data.frame(K = k.list, avg_sil = sil.k),
                aes(x = K, y = avg_sil)) +
  geom_line() +
  geom_point() +
  labs(title = "K-Means: Average Silhouette vs K",
       x = "K", y = "Average Silhouette") +
  theme_bw()
ggsave(file.path(out.dir, "M5_kmeans_silhouette_vs_K.png"),
       p.sil, width = 6, height = 5, dpi = 160)

cat("\n============================================\n")
cat("Summary metrics (you can quote in the report):\n")
cat("--------------------------------------------\n")
cat("Regression models:\n")
cat("  M1: Linear Regression  RMSE =", round(rmse.lm, 4),
    " MAE =", round(mae.lm, 4), "\n")
cat("  M2: Random Forest      RMSE =", round(rmse.rf, 4),
    " MAE =", round(mae.rf, 4), "\n\n")

cat("Classification models (CrimeLevel high/low):\n")
cat("  M3: kNN (raw features)    Acc =", round(acc_from_tab(tab.knn), 4),
    " MacroF1 =", round(macro_f1(tab.knn), 4), "\n")
cat("  M4: kNN (PCA features)    Acc =", round(acc_from_tab(tab.knn.pca), 4),
    " MacroF1 =", round(macro_f1(tab.knn.pca), 4), "\n\n")

cat("Clustering:\n")
print(crime.by.cluster)
cat("\nAll plots saved under:", normalizePath(out.dir), "\n")
cat("Use these results + plots to write the EDA, model comparison,\n")
cat("and decisions sections for the 6000-level assignment.\n")
