library(e1071)

svm.train <- function(X, y) {
  svm(x = X, y = y)
}

svm.predict <- function(X, model) {
  prob = predict(model, X)
  list(
    Prob = prob,
    Output = round(prob, 0)
  )
}