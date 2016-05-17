library(xgboost)

gboost.train <- function(X, y) {
  xgboost(data = X, 
          label = y,
          nthreads = 4,
          nrounds = 200,
          #max.depth = 50,
          #lambda = 0.5,
          verbose = 0,
          objective = "binary:logistic")
}

gboost.predict <- function(X, model) {
  prob <- xgboost::predict(model, X)
  list(
    Prob = prob,
    Output = round(prob, 0)
  )
}