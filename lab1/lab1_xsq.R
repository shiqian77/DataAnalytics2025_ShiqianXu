library(readr)
library(EnvStats)
library(nortest)

# set working directory (relative path)
#setwd("~/Courses/Data Analytics/Fall25/labs/lab 1/")
setwd("~/Desktop/lab1")

# read data
epi.data <- read_csv("epi_results_2024_pop_gdp.csv")


BDH <- epi.data$BDH.new
SPI <- epi.data$SPI.new
# print summary of variables in dataframe
summary(epi.data$BDH.new)
summary(epi.data$SPI.new)



# boxplot of variable(s)
boxplot(BDH, SPI, names = c("BDH","SPI"))


### Histograms ###

# histogram (frequency distribution) over range
hist(BDH, prob=TRUE)

# print estimated density curve for variable
lines(density(BDH,na.rm=TRUE,bw="SJ")) # or try bw=“SJ”


# histogram (frequency distribution) over rabge
hist(SPI, prob=TRUE) 

# print estimated density curve for variable
lines(density(SPI,na.rm=TRUE, bw="SJ"))




### Empirical Cumulative Distribution Function ###

# plot ecdfs
plot(ecdf(BDH), do.points=FALSE, verticals=TRUE) 

plot(ecdf(SPI), do.points=FALSE, verticals=TRUE) 


### Quantile-quantile Plots ###

# print quantile-quantile plot for variable with theoretical normal distribuion
qqnorm(BDH); qqline(BDH)

qqnorm(SPI); qqline(SPI)



# print quantile-quantile plot for 2 variables
qqplot(BDH, SPI, xlab = "Q-Q plot for BDH vs SPI") 

## Statistical Tests

#Normality statistical tests for each variable
shapiro.test(BDH)
shapiro.test(SPI)

ad.test(BDH)
ad.test(SPI)

#Statistical test for the variables having identical distributions
ks.test(BDH,SPI)

wilcox.test(BDH,SPI)

t.test(BDH,SPI)