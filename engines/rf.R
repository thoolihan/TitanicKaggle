ctrl <- trainControl(method = "repeatedcv", 
                     repeats = 3,
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE)

rf.train <- function(X, y) {
  train(x = X, 
        y = y,
        method = "rf",
        metric = "ROC",
        trControl = ctrl,        
        gridTune = data.frame(.mtry = 1:5))
}

rf.predict <- function(X, model) {
  prob <- predict(model, X)
  list(
    Prob = prob,
    Output = round(prob, 0)
  )
}