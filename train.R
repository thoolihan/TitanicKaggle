library(dplyr)
source('engines/logreg.R')
source('engines/svm.R')
source('engines/gboost.R')
source('engines/rforest.R')

test_run = TRUE
train_pct = 0.8

# read
train_raw <- read.csv("data/train.csv", header = TRUE)
test_raw <- read.csv("data/test.csv", header = TRUE)

# prepare
titanic.prepare <- function(df, median_age, median_fare) {
  row.names(df) <- df$PassengerId
  
  df <- mutate(df,
               Bias = 1,
               FirstClass = ifelse(Pclass == 1, 1, 0),
               SecondClass = ifelse(Pclass == 2, 1, 0),
               ThirdClass = ifelse(Pclass == 3, 1, 0),
               Male = ifelse(Sex == 'male', 1, 0),
               Female = ifelse(Sex == 'female', 1, 0),
               SibSp = SibSp / 10.0,
               Parch = Parch / 5.0,
               Age = ifelse(is.na(Age), median_age, Age) / 100.0,
               Fare = ifelse(is.na(Fare), median_fare, Fare) / 600.0,
               Cherbourg = ifelse(Embarked == 'C', 1, 0),
               Queenstown = ifelse(Embarked == 'Q', 1, 0),
               Southampton = ifelse(Embarked == 'S', 1, 0),
               FCW = FirstClass * Female,
               FCM = FirstClass * Male,
               SCW = SecondClass * Female,
               SCM = SecondClass * Male,
               TCW = ThirdClass * Female,
               TCM = ThirdClass * Male
  )
  
  df <- select(df, -Name, -Pclass, -Sex, -Ticket, -Cabin, -Embarked) 
  df
}

# get median values for fields that have NAs
m_age = median(train_raw$Age, na.rm = TRUE)
m_fare = median(train_raw$Fare, na.rm = TRUE)

train <- titanic.prepare(train_raw, m_age, m_fare)

# randomize
train <- train[sample(nrow(train)),]

if(test_run) { # split the training set
  split <- 0:floor(train_pct * nrow(train))
  test <- train[-(split),]
  train <- train[split,]
} else { # use the test file set
  test <- titanic.prepare(test_raw, m_age, m_fare)
}

# train
X <- data.matrix(select(train, -Survived, -PassengerId))
y <- train$Survived
yf <- as.factor(y)
logreg.model <- logreg.train(X, y)
svm.model <- svm.train(X, yf)
gboost.model <- gboost.train(X, y)
rforest.model <- rforest.train(X, yf)

# predict
if(test_run) {
  X2 <- data.matrix(select(test, -PassengerId, -Survived))
} else {
  X2 <- data.matrix(select(test, -PassengerId))
}

# add output to test set df
apply_results <- function(df, result_list) {
  df[, 'Prob'] <- result_list$Prob
  df[, 'Output'] <- result_list$Output
  df
}
logreg.test <- apply_results(test, logreg.predict(X2, logreg.model))
svm.test <- apply_results(test, svm.predict(X2, svm.model))
gboost.test <- apply_results(test, gboost.predict(X2, gboost.model))
rforest.test <- apply_results(test, rforest.predict(X2, rforest.model))

# output for test runs, write file for submission
score <- function(label, predicted) {
  tp <- sum(label == 1 & predicted == 1)
  tn <- sum(label == 0 & predicted == 0)
  fp <- sum(label == 0 & predicted == 1)
  fn <- sum(label == 1 & predicted == 0)
  
  results <- list(
    acc = round((tp + tn) / nrow(test), 3),
    prec = round(tp / (tp + fp), 3),
    rec = round(tp / (fn + tp), 3)
  )
  results$f1 = round(2 * (results$prec * results$rec) / 
                       (results$prec + results$rec), 3)
  results
}

if(test_run) {
  scores <- data.frame(acc = numeric(), 
                       prec = numeric(), 
                       rec = numeric(), 
                       f1 = numeric())
  scores['logreg',] <- score(logreg.test$Survived, logreg.test$Output)
  scores['svm',] <- score(svm.test$Survived, svm.test$Output)
  scores['gboost',] <- score(gboost.test$Survived, gboost.test$Output)  
  scores['rforest',] <- score(rforest.test$Survived, rforest.test$Output)
  arrange(scores, desc(f1))
  print(scores)
} else {
  ts = format(Sys.time(), "%Y.%m.%d.%H.%M.%S")
  write_results <- function(df, name, subdir = ts) {
    df <- mutate(df, Survived = Output) %>%
      select(PassengerId, Survived) %>%
      arrange(PassengerId)
    dir = paste('output/', subdir, sep = "")
    if(!dir.exists(dir)) {
      dir.create(dir)
    }
    fname <- paste(dir, '/', name, sep = "")
    write.csv(df, file = fname, row.names = FALSE, quote = FALSE)
    print(paste('wrote', fname))
  }
  write_results(logreg.test, 'logreg.csv')
  write_results(svm.test, 'svm.csv')
  write_results(gboost.test, 'gboost.csv')  
  write_results(rforest.test, 'rforest.csv')
}


