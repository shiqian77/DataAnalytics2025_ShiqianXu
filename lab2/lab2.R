####### Data Analytics Fall 2025 Lab 2 - Final Version ######

library(ggplot2)

### set working directory
setwd("~/Desktop/itws_6600/lab2")

### read in data
ny.data <- read.csv("NY-House-Dataset.csv", header=TRUE)



# Select only the columns needed for the analysis
columns_to_keep <- c("PRICE", "BEDS", "PROPERTYSQFT", "BATH")
ny.data.clean <- ny.data[, columns_to_keep]

# Remove rows with NA, zero, or invalid values in the selected columns
ny.data.clean <- ny.data.clean[
  !is.na(ny.data.clean$PRICE) & ny.data.clean$PRICE > 0 &
    !is.na(ny.data.clean$PROPERTYSQFT) & ny.data.clean$PROPERTYSQFT > 0 &
    !is.na(ny.data.clean$BEDS) & ny.data.clean$BEDS > 0 &
    !is.na(ny.data.clean$BATH) & ny.data.clean$BATH > 0,
]

# Apply specific filters
ny.data.clean <- ny.data.clean[ny.data.clean$PRICE < 1.95e+08, ]
ny.data.clean <- ny.data.clean[ny.data.clean$PROPERTYSQFT <= 10000, ]
ny.data.clean <- ny.data.clean[ny.data.clean$BATH == floor(ny.data.clean$BATH), ]

# Transform the response variable (PRICE) for a better model fit
ny.data.clean$log_price <- log10(ny.data.clean$PRICE)

cat(paste("Number of rows after cleaning:", nrow(ny.data.clean), "\n\n"))




### Model 1: Combination of PROPERTYSQFT and BEDS ###
model1 <- lm(log_price ~ PROPERTYSQFT + BEDS, data = ny.data.clean)

# Print summary stats for Model 1
print("--- MODEL 1 SUMMARY: Price ~ PROPERTYSQFT + BEDS ---")
print(summary(model1))

# Plot most significant variable (PROPERTYSQFT) vs. Price for Model 1
print(
  ggplot(ny.data.clean, aes(x = PROPERTYSQFT, y = log_price)) +
    geom_point(alpha = 0.2) +
    geom_smooth(method = "lm") +
    labs(title="Model 1: Price vs. Most Significant Variable (PROPERTYSQFT)",
         x="Property Square Foot", y="Log10(Price)")
)

# Plot residuals for Model 1
print(
  ggplot(model1, aes(x = .fitted, y = .resid)) +
    geom_point(alpha = 0.2) +
    geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
    labs(title='Model 1: Residual vs. Fitted Values Plot',
         x='Fitted Values (Log10 Price)', y='Residuals')
)


### Model 2: Combination of PROPERTYSQFT and BATH ###
model2 <- lm(log_price ~ PROPERTYSQFT + BATH, data = ny.data.clean)

# Print summary stats for Model 2
print("--- MODEL 2 SUMMARY: Price ~ PROPERTYSQFT + BATH ---")
print(summary(model2))

# Plot most significant variable (BATH) vs. Price for Model 2
print(
  ggplot(ny.data.clean, aes(x = BATH, y = log_price)) +
    geom_point(alpha = 0.2) +
    geom_smooth(method = "lm") +
    labs(title="Model 2: Price vs. Most Significant Variable (BATH)",
         x="Property Square Foot", y="Log10(Price)")
)

# Plot residuals for Model 2
print(
  ggplot(model2, aes(x = .fitted, y = .resid)) +
    geom_point(alpha = 0.2) +
    geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
    labs(title='Model 2: Residual vs. Fitted Values Plot',
         x='Fitted Values (Log10 Price)', y='Residuals')
)


### Model 3: Combination of PROPERTYSQFT, BEDS, and BATH ###
model3 <- lm(log_price ~ PROPERTYSQFT + BEDS + BATH, data = ny.data.clean)

# Print summary stats for Model 3
print("--- MODEL 3 SUMMARY: Price ~ PROPERTYSQFT + BEDS + BATH ---")
print(summary(model3))

# Plot most significant variable (BATH) vs. Price for Model 3
print(
  ggplot(ny.data.clean, aes(x = BATH, y = log_price)) +
    geom_point(alpha = 0.2) +
    geom_smooth(method = "lm") +
    labs(title="Model 3: Price vs. Most Significant Variable (BATH)",
         x="Property Square Foot", y="Log10(Price)")
)

# Plot residuals for Model 3
print(
  ggplot(model3, aes(x = .fitted, y = .resid)) +
    geom_point(alpha = 0.2) +
    geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
    labs(title='Model 3: Residual vs. Fitted Values Plot',
         x='Fitted Values (Log10 Price)', y='Residuals')
)
