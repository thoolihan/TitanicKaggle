library(randomForest)

rforest.train <- function(X, y) {
  randomForest(x = X,
               y = y,
               replace = TRUE,
               importance = TRUE,
               proximity = TRUE)
}

rforest.predict <- function(X, model) {
  prob = predict(model, X)
  list(
    Prob = prob,
    Output = round(prob, 0)
  )
}