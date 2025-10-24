##########################################
### Principal Component Analysis (PCA) ###
##########################################

library(caret)
library(ggfortify)
library(ggplot2)
library(class)       

# --- Load and Prepare Wine Data ---
setwd("~/Desktop/itws_6600/lab4")
wine.df <- read.csv("wine.data", header = FALSE)


names(wine.df) <- c("Type","Alcohol","Malic acid","Ash","Alcalinity of ash","Magnesium",
                    "Total phenols","Flavanoids","Nonflavanoid Phenols","Proanthocyanins",
                    "Color Intensity","Hue","Od280/od315 of diluted wines","Proline")

# Convert the class 'Type' to a factor
wine.df$Type <- as.factor(wine.df$Type)

head(wine.df)

# creating another dataframe from wine dataset
# X contains the features (columns 2-14)
# Y contains the target class (column 1)
X <- wine.df[, 2:14]
Y <- wine.df[, 1]


####### PCA #######

Xmat <- as.matrix(X)

# Center and SCALE the data
Xc <- scale(Xmat, center = T, scale = T)

# --- 1. Compute the PCs ---
principal_components <- princomp(Xc)

# Show summary of variance explained
summary(principal_components)


# --- 2. Plot dataset using 1st and 2nd PCs ---

# Combine scores and original Type for plotting
pc_scores.df <- as.data.frame(principal_components$scores)
pc_scores.df$Type <- Y

## scatter plot of dataset in PCs, colored by Type
# This plot shows the clear separation of the classes in the new PC space
print(ggplot(pc_scores.df, aes(x = Comp.1, y = Comp.2, color = Type)) + 
        geom_point(size = 2) +
        ggtitle("Wine Dataset Projected onto 1st and 2nd PCs") +
        stat_ellipse(type = "t", linetype = 2, alpha = 0.5) +
        theme_bw())

# This plot shows both the scores (points) and loadings (vectors)
print(autoplot(principal_components, data = wine.df, colour = 'Type',
               loadings = TRUE, loadings.colour = 'blue',
               loadings.label = TRUE, loadings.label.size = 3, scale = 0))


# --- 3. Identify variables that contribute the most ---
print(principal_components$loadings)


####### kNN Classification Comparison #######

# --- Train/Test split  ---
set.seed(1) 
# We use 70% of the data for training
train.idx <- sample(nrow(Xc), floor(0.7 * nrow(Xc)))

# --- Data for Model 1 (All 13 Variables) ---
# We use the SCALED data (Xc) for kNN
X_train.all <- Xc[train.idx, ]
X_test.all  <- Xc[-train.idx, ]

# --- Data for Model 2 (First 2 PCs) ---
# Get scores from PCA result
PC_scores <- principal_components$scores[, 1:2]

X_train.pc <- PC_scores[train.idx, ]
X_test.pc  <- PC_scores[-train.idx, ]

# --- Target vectors (same for both models) ---
Y_train <- Y[train.idx]
Y_test  <- Y[-train.idx]

# --- Helper functions for metrics  ---
acc_from_tab <- function(tab) sum(diag(tab)) / sum(tab)

macro_precision <- function(tab) {
  # Precision = TP / (TP + FP) = diag / rowSums
  prec <- diag(tab) / rowSums(tab)
  mean(prec, na.rm = TRUE)
}

macro_recall <- function(tab) {
  # Recall = TP / (TP + FN) = diag / colSums
  rec  <- diag(tab) / colSums(tab)
  mean(rec, na.rm = TRUE)
}

macro_f1 <- function(tab) {
  prec <- diag(tab) / rowSums(tab)
  rec  <- diag(tab) / colSums(tab)
  f1   <- 2 * prec * rec / (prec + rec)
  mean(f1, na.rm = TRUE)
}

# We use k=5 for this example
K_val <- 5 

# --- 4. Train a classifier (kNN) using all variables ---
knn.pred.all <- knn(
  train = X_train.all,
  test = X_test.all,
  cl = Y_train,
  k = K_val
)

# --- 5. Train a classifier (kNN) using first 2 PCs ---
knn.pred.pc <- knn(
  train = X_train.pc,
  test = X_test.pc,
  cl = Y_train,
  k = K_val
)

# --- 6. Compare the 2 classification models ---

# Contingency Tables
tab.all <- table(predicted = knn.pred.all, actual = Y_test)
tab.pc  <- table(predicted = knn.pred.pc, actual = Y_test)

cat("\n\n--- Model 1: kNN (k=5) with All 13 Variables ---\n")
print(tab.all)
cat("Accuracy (All Vars):     ", round(acc_from_tab(tab.all), 4), "\n")
cat("Macro Precision (All Vars):", round(macro_precision(tab.all), 4), "\n")
cat("Macro Recall (All Vars):   ", round(macro_recall(tab.all), 4), "\n")
cat("Macro F1 (All Vars):       ", round(macro_f1(tab.all), 4), "\n")

cat("\n--- Model 2: kNN (k=5) with First 2 PCs ---\n")
print(tab.pc)
cat("Accuracy (2 PCs):     ", round(acc_from_tab(tab.pc), 4), "\n")
cat("Macro Precision (2 PCs):", round(macro_precision(tab.pc), 4), "\n")
cat("Macro Recall (2 PCs):   ", round(macro_recall(tab.pc), 4), "\n")
cat("Macro F1 (2 PCs):       ", round(macro_f1(tab.pc), 4), "\n")

cat("\n--- Comparison Summary ---\n")
cat("The model using all 13 variables performed worse than the model using only the first 2 PCs, because accuracy, precision, recall F1 of first 2 PCs are all higher than that using all 13 variables.", "\n")

