library(xgboost)
library(dplyr)

titanic.train <- function(X, y) {
  xgboost(data = X, 
          label = y,
          nthreads = 4,
          nrounds = 200,
          #max.depth = 50,
          #lambda = 0.5,
          objective = "binary:logistic")
}

titanic.predict <- function(df, model) {
  X = data.matrix(df)
  df$Prob <- xgboost::predict(model, X)
  df$Output <- round(df$Prob, 0)
  df
}