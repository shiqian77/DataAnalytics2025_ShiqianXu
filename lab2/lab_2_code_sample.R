####### Data Analytics Fall 2025 Lab 1 ######

library(ggplot2)

### set working directory
setwd("~/Desktop/itws_6600/lab2")

### read in data
ny_house.data <- read.csv("NY-House-Dataset.csv", header=TRUE)

View(ny_house.data)

#### Exploratory Analysis ####

price <- ny_house.data$PRICE

## NA values
na.indices <- is.na(EPI.new) 

## drop NAs
Epi.new.compl <- EPI.new[!na.indices]

## convert to data frame and add country
country <- epi.data$country

Epi.new.compl <- data.frame(Country = country[!na.indices], EPI = EPI.new[!na.indices])

## summary stats
summary(EPI.new)

fivenum(EPI.new,na.rm=TRUE)

## histograms
hist(EPI.new)

hist(EPI.new, seq(20., 80., 2.0), prob=TRUE)

rug(EPI.new)

lines(density(EPI.new,na.rm=TRUE,bw=1))
lines(density(EPI.new,na.rm=TRUE,bw="SJ"))

##################

### Comparing distributions of 2 variables
EPI.old <- epi.data$EPI.old

boxplot(EPI.old, EPI.new, names=c("EPI.old","EPI.new"))


### Quantile-quantile plots

qqplot(EPI.new,EPI.old)


#### GDP vs. EPI ####
gdp <- epi.data$gdp

ggplot(epi.data, aes(x = gdp, y = EPI.new, colour = region)) +
  geom_point()

## created linear model of EPI.new ~ gdp
lin.mod0 <- lm(price~PropertySqFt+ Beds+ Bath,ny_house.data)

summary(lin.mod0)

ggplot(ny_house.data, aes(x = PropertySqFt+ Beds+ Bath, y = price)) +
  geom_point() +
  stat_smooth(method = "lm")

ggplot(lin.mod0, aes(x = .fitted, y = .resid)) +
  geom_point() +
  geom_hline(yintercept = 0) +
  labs(title='Residual vs. Fitted Values Plot', x='Fitted Values', y='Residuals')


## another lm using log 10 gdp
ny_house.data$ PropertySqFt+ Beds+ Bath<- log10(PropertySqFt+ Beds+ Bath)

ggplot(epi.data, aes(x = log_gdp, y = EPI.new, colour = region)) +
  geom_point()
#ggplot(epi.data, aes(x = log10(population), y = EPI.new, colour = region)) +
 # geom_point()
lin.mod1 <- lm(EPI.new~log_gdp+population,epi.data)
#lin.mod1 <- lm(EPI.new~log_gdp+log10(population),epi.data)
summary(lin.mod1)

ggplot(epi.data, aes(x = log_gdp, y = EPI.new)) +
  geom_point() +
  stat_smooth(method = "lm")

ggplot(lin.mod1, aes(x = .fitted, y = .resid)) +
  geom_point() +
  geom_hline(yintercept = 0) +
  labs(title='Residual vs. Fitted Values Plot', x='Fitted Values', y='Residuals')

### subset by regions
summary(epi.data$region)

##convert region from strings to factors
epi.data$region <- as.factor(epi.data$region)

summary(epi.data$region)

#data cleaning, exclude the part you don't want
ny_house.data.subset <- ny_house.data[! ny_house.data$region %in% c("Greater Middle East","Global West"),c("gdp","country")]

ggplot(epi.data.subset, aes(x = log_gdp, y = EPI.new, colour = region, label=country)) +
  geom_point() + geom_text(hjust=0, vjust=0)

lin.mod2 <- lm(EPI.new~log_gdp,epi.data.subset)

summary(lin.mod2)

ggplot(epi.data.subset, aes(x = log_gdp, y = EPI.new)) +
  geom_point() +
  stat_smooth(method = "lm")

ggplot(lin.mod2, aes(x = .fitted, y = .resid)) +
  geom_point() +
  geom_hline(yintercept = 0) +
  labs(title='Residual vs. Fitted Values Plot', x='Fitted Values', y='Residuals')

