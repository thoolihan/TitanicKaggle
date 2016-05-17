library(xgboost)
library(dplyr)

titanic.train <- function(df) {
  X <- data.matrix(select(df, -Survived))
  y <- df$Survived
  xgboost(data = X, 
          label = y,
          nrounds = 10,
          max.depth = 10,
          lambda = 2,
          objective = "binary:logistic")
}

titanic.predict <- function(df, model) {
  df$Prob <- xgboost::predict(model, data.matrix(df))
  df$Output <- round(df$Prob, 0)
  df
}