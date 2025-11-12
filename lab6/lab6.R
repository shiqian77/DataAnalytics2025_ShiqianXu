################################
# Evaluating Regression Models #
################################

setwd("~/Desktop/itws_6600/lab6")

library(readr)
library(ggplot2)
library(e1071)
library(caret)
library(cv)
library(broom)

## Output directory 
out_dir <- "plots"
if (!dir.exists(out_dir)) dir.create(out_dir)

NY_House_Dataset <- read_csv("NY-House-Dataset.csv")
dataset <- NY_House_Dataset
names(dataset)

## Plot
p1 <- ggplot(dataset, aes(x = PROPERTYSQFT, y = PRICE)) + geom_point()
ggsave(file.path(out_dir, "01_scatter_raw.png"), p1, width = 6, height = 4, dpi = 300)

p2 <- ggplot(dataset, aes(x = log10(PROPERTYSQFT), y = log10(PRICE))) + geom_point()
ggsave(file.path(out_dir, "02_scatter_loglog.png"), p2, width = 6, height = 4, dpi = 300)

## Linear Regression (full) 
lin.mod0 <- lm(log10(PRICE) ~ log10(PROPERTYSQFT), dataset)
summary(lin.mod0)

p3 <- ggplot(dataset, aes(x = log10(PROPERTYSQFT), y = log10(PRICE))) +
  geom_point() + stat_smooth(method = "lm", col = "blue")
ggsave(file.path(out_dir, "03_lm_full_fit.png"), p3, width = 6, height = 4, dpi = 300)

aug0 <- augment(lin.mod0)
p4 <- ggplot(aug0, aes(x = .fitted, y = .resid)) +
  geom_point() + geom_hline(yintercept = 0)
ggsave(file.path(out_dir, "04_lm_full_residuals.png"), p4, width = 6, height = 4, dpi = 300)

## Light cleaning 
dataset.sub0 <- dataset[-which(dataset$PROPERTYSQFT == 2184.207862 | dataset$PRICE > 20000000),]

## Linear model on cleaned 
lin.mod1 <- lm(log10(PRICE) ~ log10(PROPERTYSQFT), dataset.sub0)
summary(lin.mod1)

p5 <- ggplot(dataset.sub0, aes(x = log10(PROPERTYSQFT), y = log10(PRICE))) +
  geom_point() + stat_smooth(method = "lm", col = "blue")
ggsave(file.path(out_dir, "05_lm_clean_fit.png"), p5, width = 6, height = 4, dpi = 300)

aug1 <- augment(lin.mod1)
p6 <- ggplot(aug1, aes(x = .fitted, y = .resid)) +
  geom_point() + geom_hline(yintercept = 0)
ggsave(file.path(out_dir, "06_lm_clean_residuals.png"), p6, width = 6, height = 4, dpi = 300)

## SVM (linear) 
svm.mod_lin <- svm(log10(PRICE) ~ log10(PROPERTYSQFT), dataset.sub0, kernel = "linear")
svm.pred_lin <- data.frame(real = log10(dataset.sub0$PRICE),
                           predicted = predict(svm.mod_lin, dataset.sub0))

p7 <- ggplot(dataset.sub0, aes(x = log10(PROPERTYSQFT), y = log10(PRICE))) +
  geom_point() + geom_line(aes(y = svm.pred_lin$predicted), col = "green")
ggsave(file.path(out_dir, "07_svm_linear_fit.png"), p7, width = 6, height = 4, dpi = 300)

p8 <- ggplot(svm.pred_lin, aes(x = predicted, y = real - predicted)) +
  geom_point() + geom_hline(yintercept = 0)
ggsave(file.path(out_dir, "08_svm_linear_residuals.png"), p8, width = 6, height = 4, dpi = 300)

## SVM (radial)
gamma.range <- 10^seq(-3, 2, 1)
C.range     <- 10^seq(-3, 2, 1)

tuned.svm <- tune.svm(log10(PRICE) ~ log10(PROPERTYSQFT),
                      data = dataset.sub0,
                      kernel = "radial",
                      gamma = gamma.range,
                      cost  = C.range,
                      tune.control = tune.control(cross = 5))

opt.gamma <- tuned.svm$best.parameters$gamma
opt.C     <- tuned.svm$best.parameters$cost

svm.mod_rbf <- svm(log10(PRICE) ~ log10(PROPERTYSQFT),
                   dataset.sub0,
                   kernel = "radial",
                   gamma  = opt.gamma,
                   cost   = opt.C)

svm.pred_rbf <- data.frame(real = log10(dataset.sub0$PRICE),
                           predicted = predict(svm.mod_rbf, dataset.sub0))

p9 <- ggplot(dataset.sub0, aes(x = log10(PROPERTYSQFT), y = log10(PRICE))) +
  geom_point() + geom_line(aes(y = svm.pred_rbf$predicted), col = "red")
ggsave(file.path(out_dir, "09_svm_rbf_fit.png"), p9, width = 6, height = 4, dpi = 300)

p10 <- ggplot(svm.pred_rbf, aes(x = predicted, y = real - predicted)) +
  geom_point() + geom_hline(yintercept = 0)
ggsave(file.path(out_dir, "10_svm_rbf_residuals.png"), p10, width = 6, height = 4, dpi = 300)

## Single split metrics 
set.seed(2025)
train.indexes <- sample(nrow(dataset.sub0), 0.75 * nrow(dataset.sub0))
train <- dataset.sub0[train.indexes, ]
test  <- dataset.sub0[-train.indexes, ]

lin.split <- lm(log10(PRICE) ~ log10(PROPERTYSQFT), train)
cv.res <- cv(lin.split)
summary(cv.res)

lm.pred0 <- predict(lin.split, test)
err <- lm.pred0 - log10(test$PRICE)
mae <- mean(abs(err)); mse <- mean(err^2); rmse <- sqrt(mean(err^2))
cat("Single-split LM  MAE/MSE/RMSE:",
    round(mae, 4), round(mse, 4), round(rmse, 4), "\n")

## Monte Carlo CV 
k <- 100
set.seed(2025)

mae0 <- mse0 <- rmse0 <- numeric(k)
mae1 <- mse1 <- rmse1 <- numeric(k)
mae2 <- mse2 <- rmse2 <- numeric(k)

for (i in 1:k) {
  idx   <- sample(nrow(dataset.sub0), 0.75 * nrow(dataset.sub0))
  train <- dataset.sub0[idx, ]
  test  <- dataset.sub0[-idx, ]
  
  # Linear regression
  m0 <- lm(log10(PRICE) ~ log10(PROPERTYSQFT), train)
  p0 <- predict(m0, test)
  e0 <- p0 - log10(test$PRICE)
  mae0[i] <- mean(abs(e0))
  mse0[i] <- mean(e0^2)
  rmse0[i] <- sqrt(mean(e0^2))
  
  # Linear SVM
  m1 <- svm(log10(PRICE) ~ log10(PROPERTYSQFT), train, kernel = "linear")
  p1 <- predict(m1, test)
  e1 <- p1 - log10(test$PRICE)
  mae1[i] <- mean(abs(e1))
  mse1[i] <- mean(e1^2)
  rmse1[i] <- sqrt(mean(e1^2))
  
  # RBF SVM (re-use global tuned gamma/C; no per-split tuning)
  m2 <- svm(log10(PRICE) ~ log10(PROPERTYSQFT),
            data = train, kernel = "radial",
            gamma = opt.gamma, cost = opt.C)
  p2 <- predict(m2, test)
  e2 <- p2 - log10(test$PRICE)
  mae2[i] <- mean(abs(e2))
  mse2[i] <- mean(e2^2)
  rmse2[i] <- sqrt(mean(e2^2))
}

## Results 
results <- data.frame(
  Model = c("Linear Regression", "Linear SVM", "RBF SVM (tuned, global params)"),
  MAE   = c(mean(mae0), mean(mae1), mean(mae2)),
  MSE   = c(mean(mse0), mean(mse1), mean(mse2)),
  RMSE  = c(mean(rmse0), mean(rmse1), mean(rmse2))
)
print(results)

cat("\nAll plots saved under:", normalizePath(out_dir), "\n")