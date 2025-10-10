#ex1
library(class)

# --- Load data  ---
setwd("~/Desktop/itws_6600/lab3")

abalone.data <- read.csv("abalone_dataset.csv")

# --- Create target: age.group from rings (do NOT use rings as predictor) ---
abalone.data$age.group <- cut(
  abalone.data$rings,
  br = c(0, 8, 11, 35),
  labels = c("young", "adult", "old")
)
abalone.data$age.group <- factor(abalone.data$age.group)


# --- Allowed predictors: all columns EXCEPT sex, rings, age.group ---
all.features <- setdiff(names(abalone.data), c("sex", "rings", "age.group"))

abalone.data[, all.features] <- scale(abalone.data[, all.features])

nm <- names(all.features)
nm[nm == "shucked_wieght"] <- "shucked_weight"
nm[nm == "viscera_wieght"] <- "viscera_weight"
names(all.features) <- nm

# Two different subsets (both exclude sex & rings)
feats.A <- intersect(c("length", "diameter", "height"), all.features) 
feats.B <- intersect(c("whole_weight", "shucked_weight", "viscera_weight", "shell_weight"), all.features) 

# --- Train/Test split (exact File 3 style: fixed 100 for training) ---
set.seed(1)                         
s.train <- sample(nrow(abalone.data), 3000)


abalone.train <- abalone.data[s.train, ]
abalone.test  <- abalone.data[-s.train, ]

dim(abalone.test)              
dim(abalone.train)

# --- kNN with k = 50 for both subsets ---
k0 <- 50

# --- Model A (size features) ---
knn.predicted.A <- knn(
  abalone.train[, feats.A],
  abalone.test[,  feats.A],
  abalone.train$age.group,
  k = k0
)
tab.A <- table(knn.predicted.A, abalone.test$age.group,
               dnn = c("predicted","actual"))
acc.A <- sum(diag(tab.A)) / sum(tab.A)

cat("\n=== kNN Model A (", paste(feats.A, collapse = ", "),
    "; k = ", k0, ") ===\n", sep = "")
print(tab.A)
cat("Accuracy A: ", round(acc.A, 4), "\n", sep = "")

# --- Model B (weight features) ---
knn.predicted.B <- knn(
  abalone.train[, feats.B],
  abalone.test[,  feats.B],
  abalone.train$age.group,
  k = k0
)
tab.B <- table(knn.predicted.B, abalone.test$age.group,
               dnn = c("predicted","actual"))
acc.B <- sum(diag(tab.B)) / sum(tab.B)

cat("\n=== kNN Model B (", paste(feats.B, collapse = ", "),
    "; k = ", k0, ") ===\n", sep = "")
print(tab.B)
cat("Accuracy B: ", round(acc.B, 4), "\n", sep = "")

# --- Pick better model and tune k (still excluding sex & rings) ---
if (acc.A >= acc.B) {
  best.feats <- feats.A
  best.name  <- "A (size features)"
} else {
  best.feats <- feats.B
  best.name  <- "B (weight features)"
}

k.grid <- 1:100
acc.grid <- numeric(length(k.grid))

for (i in seq_along(k.grid)) {
  pred <- knn(
    abalone.train[, best.feats],
    abalone.test[,  best.feats],
    abalone.train$age.group,
    k = k.grid[i]
  )
  tab <- table(pred, abalone.test$age.group)
  acc.grid[i] <- sum(diag(tab)) / sum(tab)
}

best.k   <- k.grid[which.max(acc.grid)]
best.acc <- max(acc.grid)

cat("\n=== k tuning for best model: ", best.name, " ===\n", sep = "")
cat("Best k: ", best.k, "\n", sep = "")
cat("Best tuned accuracy: ", round(best.acc, 4), "\n", sep = "")

#ex2 helper code
#############################################
##### Classification: Abalone with kNN #####
#############################################

library(class)

# --- Load data  ---
abalone.data <- read.csv("abalone_dataset.csv")


all.features <- setdiff(names(abalone.data), c("sex", "rings", "age.group"))

abalone.data[, all.features] <- scale(abalone.data[, all.features])

nm <- names(abalone.data)
# common misspellings
nm[nm == "shucked_wieght"] <- "shucked_weight"
nm[nm == "viscera_wieght"] <- "viscera_weight"
names(abalone.data) <- nm

# --- Create target: age.group from rings (do NOT use rings as predictor) ---
abalone.data$age.group <- cut(
  abalone.data$rings,
  br = c(0, 8, 11, 35),
  labels = c("young", "adult", "old")
)
abalone.data$age.group <- factor(abalone.data$age.group)

#  --- Clean rows if needed ---
abalone.data <- na.omit(abalone.data)

# --- Allowed predictors: all columns EXCEPT sex, rings, age.group ---
all.features <- setdiff(names(abalone.data), c("sex", "rings", "age.group"))

# Two different subsets (both exclude sex & rings)
feats.A <- intersect(c("length", "diameter", "height"), all.features) 
feats.B <- intersect(c("whole_weight", "shucked_weight", "viscera_weight", "shell_weight"), all.features)  

# --- Train/Test split ---
set.seed(1)                       
s.train <- sample(nrow(abalone.data), 3000)

abalone.train <- abalone.data[s.train, ]
abalone.test  <- abalone.data[-s.train, ]

dim(abalone.test)                 
dim(abalone.train)

# --- kNN with k = 50 for both subsets ---
k0 <- 50

# --- Model A (size features) ---
knn.predicted.A <- knn(
  abalone.train[, feats.A],
  abalone.test[,  feats.A],
  abalone.train$age.group,
  k = k0
)
tab.A <- table(knn.predicted.A, abalone.test$age.group,
               dnn = c("predicted","actual"))
acc.A <- sum(diag(tab.A)) / sum(tab.A)

cat("\n=== kNN Model A (", paste(feats.A, collapse = ", "),
    "; k = ", k0, ") ===\n", sep = "")
print(tab.A)
cat("Accuracy A: ", round(acc.A, 4), "\n", sep = "")

# --- Model B (weight features) ---
knn.predicted.B <- knn(
  abalone.train[, feats.B],
  abalone.test[,  feats.B],
  abalone.train$age.group,
  k = k0
)
tab.B <- table(knn.predicted.B, abalone.test$age.group,
               dnn = c("predicted","actual"))
acc.B <- sum(diag(tab.B)) / sum(tab.B)

cat("\n=== kNN Model B (", paste(feats.B, collapse = ", "),
    "; k = ", k0, ") ===\n", sep = "")
print(tab.B)
cat("Accuracy B: ", round(acc.B, 4), "\n", sep = "")



acc_from_tab <- function(tab) sum(diag(tab)) / sum(tab)
balanced_acc <- function(tab) {
  rec <- diag(tab) / colSums(tab)  
  mean(rec, na.rm = TRUE)
}
macro_f1 <- function(tab) {
  prec <- diag(tab) / rowSums(tab)
  rec  <- diag(tab) / colSums(tab)
  f1   <- 2 * prec * rec / (prec + rec)
  mean(f1, na.rm = TRUE)
}


eval_subset <- function(feats, k.grid = 1:100) {
  acc  <- numeric(length(k.grid))
  for (i in seq_along(k.grid)) {
    pred <- knn(abalone.train[, feats], abalone.test[, feats],
                abalone.train$age.group, k = k.grid[i])
    tab  <- table(pred, abalone.test$age.group)
    acc[i]  <- acc_from_tab(tab)
  }
  list(best_k_acc  = k.grid[which.max(acc)],
       best_acc    = max(acc),
       acc = acc)
}


resA <- eval_subset(feats.A, 1:100)
resB <- eval_subset(feats.B, 1:100)

cat("\nSubset A (", paste(feats.A, collapse=", "), ")\n", sep="")
cat("  best acc :", round(resA$best_acc,4), " at k=", resA$best_k_acc,  "\n", sep="")

cat("\nSubset B (", paste(feats.B, collapse=", "), ")\n", sep="")
cat("  best acc :", round(resB$best_acc,4), " at k=", resB$best_k_acc,  "\n", sep="")


if (resA$best_acc >= resB$best_acc) {
  best_subset <- "A"
  best_feats  <- feats.A
  best_k      <- resA$best_k_acc
} else {
  best_subset <- "B"
  best_feats  <- feats.B
  best_k      <- resB$best_k_acc
}
cat("\n>>> Best subset:", best_subset, 
    "with k =", best_k, "(by accuracy)\n")

best.feats <- best_feats

#ex2
library(ggplot2)
library(cluster)
library(factoextra)

# --- Load data ---
abalone.data <- read.csv("abalone_dataset.csv")

# --- Canonicalize weight column names (fix typos & dot variants) ---
nm <- names(abalone.data)
# common misspellings
nm[nm == "shucked_wieght"] <- "shucked_weight"
nm[nm == "viscera_wieght"] <- "viscera_weight"
names(abalone.data) <- nm

# --- Best feature subset (no sex / rings) ---
best.feats <- c("whole_weight","shucked_weight","viscera_weight","shell_weight")
stopifnot(all(best.feats %in% names(abalone.data)))

# --- Build feature matrix for clustering ---
X <- abalone.data[, best.feats]

X <- scale(X)

# --- Precompute distances (used for silhouettes) ---
D <- dist(X)


# ====== K-Means Clustering =====

k.list <- 2:10                            
# --- average silhouette by K ---
si.km  <- numeric(length(k.list))           


for (i in seq_along(k.list)) {
  k <- k.list[i]
  set.seed(100 + k) 
  km <- kmeans(X, centers = k)              
  sil <- silhouette(km$cluster, D)
  si.km[i] <- mean(sil[, 3])
}

# --- Pick best K by highest average silhouette ---
best.k.km <- k.list[which.max(si.km)]
cat("Best K for K-Means by avg silhouette:", best.k.km, "\n")

# --- Fit final K-Means at best K and plot silhouette ---
set.seed(100 + best.k.km)  
km.best <- kmeans(X, centers = best.k.km)
sil.km  <- silhouette(km.best$cluster, D)
fviz_silhouette(sil.km)

# --- Plot K vs average silhouette (diagnostic curve) ---
plot(k.list, si.km, type = "b",
     xlab = "K", ylab = "Average silhouette (K-Means)",
     main = "K-Means: Silhouette vs K")


# ====== PAM Clustering =====

si.pam <- numeric(length(k.list)) 

for (i in seq_along(k.list)) {
  k <- k.list[i]
  pam.fit <- pam(X, k)
  sil <- silhouette(pam.fit$cluster, D)
  si.pam[i] <- mean(sil[, 3])
  #print(sil)
}

# --- Pick best K by highest average silhouette ---
best.k.pam <- k.list[which.max(si.pam)]
cat("Best K for PAM by avg silhouette:", best.k.pam, "\n")

# --- Fit final PAM at best K and plot silhouette ---
pam.best <- pam(X, best.k.pam)
sil.pam  <- silhouette(pam.best$cluster, D)
fviz_silhouette(sil.pam)

# --- Plot K vs average silhouette (diagnostic curve) ---
plot(k.list, si.pam, type = "b",
     xlab = "K", ylab = "Average silhouette (PAM)",
     main = "PAM: Silhouette vs K")


