# ============================================================
# Football Goal Prediction Analysis
# ============================================================

# -------------------------
# 1. Load Packages
# -------------------------
library(tidyverse)
library(caret)
library(ranger)
library(gbm)
library(yardstick)
library(corrplot)
library(ggplot2)

set.seed(123)

# -------------------------
# 2. Data Preparation
# -------------------------
# Load dataset (file should be in the same directory as this script)
allplayerdata <- read.csv("~/2021-2022 Football Player Stats.csv", sep=";", comment.char="#")


df <- allplayerdata

df <- df %>%
  filter(Min >= 900) %>%
  mutate(TotalGoals = X90s * Goals) %>%
  select(Player, Min, Pos, TotalGoals, Goals, Shots, SoT., Assists, ShoDist, PKatt) %>%
  drop_na()

df$TotalGoals <- round(df$TotalGoals)

df_ml <- df %>%
  select(-Player, -Goals)

df_ml$Pos <- factor(df_ml$Pos)

# ============================================================
# CHAPTER 2 FIGURES
# ============================================================

# Figure 2.1
hist(df_ml$TotalGoals,
     breaks = seq(-0.5, max(df_ml$TotalGoals, na.rm = TRUE) + 0.5, by = 1),
     col = "lightgray", border = "white",
     xlab = "Seasonal goal totals",
     ylab = "Number of players",
     main = "Distribution of seasonal goal totals")

# Figure 2.2
plot(df_ml$Shots, df_ml$TotalGoals,
     xlab = "Total shots per 90 minutes",
     ylab = "Seasonal goal totals",
     pch = 16, cex = 0.7)
lines(lowess(df_ml$Shots, df_ml$TotalGoals), lwd = 2)

# ============================================================
# CHAPTER 3 FIGURES
# ============================================================

# Figure 3.1
model <- lm(TotalGoals ~ Shots, data = df_ml)

plot(df_ml$Shots, df_ml$TotalGoals,
     pch = 16, cex = 0.7,
     xlab = "Shots per 90 minutes",
     ylab = "Seasonal goal totals")

abline(model, col = "red", lwd = 2)
abline(h = 0, lty = 2)

plot(fitted(model), resid(model),
     xlab = "Fitted values",
     ylab = "Residuals",
     pch = 16, cex = 0.7)
abline(h = 0, lty = 2)

# ============================================================
# CHAPTER 4 FIGURES
# ============================================================

# Figure 4.1
p <- ggplot(df_ml, aes(x = Shots, y = TotalGoals)) +
  geom_point(alpha = 0.4) +
  labs(title = "Variation in Linear Model Fits Across Random Train/Test Splits",
       x = "Shots",
       y = "Total Goals")

for(i in 1:10){
  train_index <- sample(1:nrow(df_ml), size = 0.3 * nrow(df_ml))
  train <- df_ml[train_index, ]
  model <- lm(TotalGoals ~ Shots, data = train)
  
  p <- p + geom_abline(
    slope = coef(model)[2],
    intercept = coef(model)[1],
    alpha = 0.4
  )
}
p

# Figure 4.2 (Bootstrap example)
B <- 1000
boot_means <- numeric(B)

for (b in 1:B) {
  boot_sample <- df %>% slice_sample(n = nrow(df), replace = TRUE)
  boot_means[b] <- mean(boot_sample$TotalGoals)
}

ggplot(data.frame(boot_means), aes(x = boot_means)) +
  geom_histogram(bins = 30, colour = "black", fill = "lightgrey") +
  labs(title = "Bootstrap Distribution of Mean Seasonal Goals",
       x = "Mean seasonal goals",
       y = "Frequency")

# Figure 4.3
complexity <- seq(1,10,0.1)
bias2 <- exp(-complexity/2)
variance <- complexity/10
noise <- 0.1
test_error <- bias2 + variance + noise

plot(complexity,bias2,type="l",col="blue",ylim=c(0,1),
     ylab="Error",xlab="Model Complexity",lwd=2)
lines(complexity,variance,col="red",lwd=2)
lines(complexity,test_error,col="black",lwd=2)

# ============================================================
# CHAPTER 5: PREDICTOR ANALYSIS
# ============================================================

vars <- c("Min", "Shots", "SoT.", "Assists", "ShoDist", "PKatt")

df_plot <- df_ml[, c("TotalGoals", vars)] %>% na.omit()

X_std <- scale(df_plot[, vars])
df_std <- as.data.frame(X_std)
df_std$TotalGoals <- df_plot$TotalGoals

cutoff <- median(df_std$TotalGoals)
df_std$GoalGroup <- ifelse(df_std$TotalGoals > cutoff, "High goals", "Low goals")

cols <- ifelse(df_std$GoalGroup == "High goals", "red", "green")

# Figure 5.1
pairs(df_std[, vars],
      pch = 16, cex = 0.55,
      col = adjustcolor(cols, alpha.f = 0.5))

# Figure 5.2
par(mfrow = c(2, 3))
for (v in vars) {
  boxplot(df_std[[v]] ~ df_std$GoalGroup,
          col = c("lightgreen", "lightcoral"),
          main = v)
}
par(mfrow = c(1, 1))

# Figure 5.3
cor_mat <- cor(df_plot[, vars])
corrplot(cor_mat)

# ============================================================
# MODEL FITTING & CROSS VALIDATION
# ============================================================

k <- 10
folds <- sample(rep(1:k, length.out = nrow(df_ml)))

rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2))
}

bagged_poisson_predict <- function(train_data, new_data, B = 100) {
  preds <- matrix(NA, nrow = nrow(new_data), ncol = B)
  
  for (b in 1:B) {
    idx <- sample(seq_len(nrow(train_data)), replace = TRUE)
    boot_data <- train_data[idx, ]
    
    model <- glm(TotalGoals ~ ., family = poisson(link = "log"), data = boot_data)
    
    preds[, b] <- predict(model, newdata = new_data, type = "response")
  }
  
  rowMeans(preds)
}

lm_cv_preds    <- rep(NA, nrow(df_ml))
pois_cv_preds  <- rep(NA, nrow(df_ml))
bag_cv_preds   <- rep(NA, nrow(df_ml))
gbm_cv_preds   <- rep(NA, nrow(df_ml))

lm_cv_rmse     <- numeric(k)
pois_cv_rmse   <- numeric(k)
bag_cv_rmse    <- numeric(k)
gbm_cv_rmse    <- numeric(k)

n_trees <- 3000
depth <- 3
shrinkage <- 0.01
min_obs <- 10
B <- 100

for (i in 1:k) {
  
  train <- df_ml[folds != i, ]
  test  <- df_ml[folds == i, ]
  
  lm_fit <- lm(TotalGoals ~ ., data = train)
  lm_preds <- predict(lm_fit, newdata = test)
  lm_cv_preds[folds == i] <- lm_preds
  lm_cv_rmse[i] <- rmse(test$TotalGoals, lm_preds)
  
  pois_fit <- glm(TotalGoals ~ ., family = poisson(link = "log"), data = train)
  pois_preds <- predict(pois_fit, newdata = test, type = "response")
  pois_cv_preds[folds == i] <- pois_preds
  pois_cv_rmse[i] <- rmse(test$TotalGoals, pois_preds)
  
  bag_preds <- bagged_poisson_predict(train, test, B = B)
  bag_cv_preds[folds == i] <- bag_preds
  bag_cv_rmse[i] <- rmse(test$TotalGoals, bag_preds)
  
  gbm_fit <- gbm(
    TotalGoals ~ .,
    data = train,
    distribution = "poisson",
    n.trees = n_trees,
    interaction.depth = depth,
    shrinkage = shrinkage,
    n.minobsinnode = min_obs,
    bag.fraction = 0.7,
    train.fraction = 1.0,
    verbose = FALSE
  )
  
  gbm_preds <- predict(
    gbm_fit,
    newdata = test,
    n.trees = n_trees,
    type = "response"
  )
  
  gbm_cv_preds[folds == i] <- gbm_preds
  gbm_cv_rmse[i] <- rmse(test$TotalGoals, gbm_preds)
}

# Results
results <- data.frame(
  Model = c("Linear LM", "Poisson GLM", "Bagged Poisson GLM", "Poisson GBM"),
  CV_RMSE = c(mean(lm_cv_rmse),
              mean(pois_cv_rmse),
              mean(bag_cv_rmse),
              mean(gbm_cv_rmse))
)

print(results)

# Figure 5.4
par(mfrow = c(2, 2))

plot(df_ml$TotalGoals, lm_cv_preds, pch = 16)
abline(0, 1, col = "red")

plot(df_ml$TotalGoals, pois_cv_preds, pch = 16)
abline(0, 1, col = "red")

plot(df_ml$TotalGoals, bag_cv_preds, pch = 16)
abline(0, 1, col = "red")

plot(df_ml$TotalGoals, gbm_cv_preds, pch = 16)
abline(0, 1, col = "red")

# Dispersion
deviance(pois_fit) / df.residual(pois_fit)
