library(rpart)

rp.train <- function(X, y) {
  dftemp <- data.frame(X)
  dftemp$Survived = y
  rpart(Survived ~ ., data = dftemp)
}

rp.predict <- function(X, model) {
  prob = predict(model, data.frame(X))
  list(
    Prob = prob,
    Output = round(prob, 0)
  )
}