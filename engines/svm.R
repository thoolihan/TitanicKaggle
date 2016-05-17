library(e1071)

svm.train <- function(X, y) {
  svm(x = X, 
      y = y,
      kernel = "polynomial")
}

svm.predict <- function(X, model) {
  prob = predict(model, X)
  list(
    Prob = prob,
    Output = prob
  )
}