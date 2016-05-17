library(functional)

sigmoid <- function(z) {
  1 / (1 + exp(-z))
}

logreg.train <- function(X, y) {
  lambda <- 1
  
  cost <- function(theta, X, y, lambda) {
    vals <- X %*% theta
    (1 / nrow(X)) * sum((-1 * y) * log(sigmoid(vals)) - 
                          (1 - y) * log(1 - sigmoid(vals))) +
      ((lambda / nrow(X)) * sum(theta[2:length(theta)] ^ 2))
  }
  
  cost_wrapper <- Curry(cost, X = X, lambda = lambda)
  
  grad <- function(theta, X, y, lambda) {
    vals <- X %*% theta
    (1 / nrow(X)) * (t(X) %*% (sigmoid(vals) - y)) + 
      ((lambda / nrow(X)) * sum(theta[2:length(theta)]))
  }
  
  grad_wrapper <- Curry(grad, X = X, lambda = lambda)
  
  optim(par = rep(0, length(X[1,])),
        fn = cost_wrapper,
        gr = grad_wrapper,
        y = y,
        control = list(maxit = 1000,
                       alpha = 10))
}

logreg.predict <- function(X, model) {
  prob <- sigmoid(X %*% model$par)
  list(
    Prob = prob,
    Output = ifelse(prob > 0.5, 1, 0)
  )
}