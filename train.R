library(dplyr)
source('gboost.R')
source('logreg.R')
source('rforest.R')

test_run = TRUE
train_pct = 0.7

# read
train_raw <- read.csv("data/train.csv", header = TRUE)
test_raw <- read.csv("data/test.csv", header = TRUE)

# prepare
titanic.median_age <- function(vec) {
  median(vec, na.rm = TRUE)
}

titanic.prepare <- function(df, median_age) {
  row.names(df) <- df$PassengerId
  
  df <- mutate(df,
               Bias = 1,
               Survived = Survived,
               FirstClass = ifelse(Pclass == 1, 1, 0),
               SecondClass = ifelse(Pclass == 2, 1, 0),
               ThirdClass = ifelse(Pclass == 3, 1, 0),
               Male = ifelse(Sex == 'male', 1, 0),
               Female = ifelse(Sex == 'female', 1, 0),
               SibSp = SibSp / 10.0,
               Parch = Parch / 5.0,
               Age = ifelse(is.na(Age), median_age, Age) / 100.0,
               Fare = Fare / 600.0,
               Cherbourg = ifelse(Embarked == 'C', 1, 0),
               Queenstown = ifelse(Embarked == 'Q', 1, 0),
               Southampton = ifelse(Embarked == 'S', 1, 0),
               FCW = FirstClass * Female,
               FCM = FirstClass * Male,
               SCW = SecondClass * Female,
               SCM = SecondClass * Male,
               TCW = SecondClass * Female,
               TCM = ThirdClass * Male
  )
  
  df <- select(df, -Name, -Pclass, -Sex, -Ticket, -Cabin, -Embarked, -PassengerId) 
  df
}

m_age = titanic.median_age(train_raw$Age)

train <- titanic.prepare(train_raw, m_age)

# randomize
train <- train[sample(nrow(train)),]

if(test_run) {
  split = 0:floor(train_pct * nrow(train))
  test = train[-(split),]
  train = train[split,]
} else {
  test <- titanic.prepare(test_raw, m_age)
}

# train
X = data.matrix(select(train, -Survived))
y = train$Survived
gboost.model = gboost.train(X, y)
logreg.model = logreg.train(X, y)
rforest.model = rforest.train(X, y)

# predict
X2 = data.matrix(select(test, -Survived))
apply_results <- function(df, result_list) {
  df[, 'Prob'] <- result_list$Prob
  df[, 'Output'] <- result_list$Output
  df
}
gboost.test <- apply_results(test, gboost.predict(X2, gboost.model))
logreg.test <- apply_results(test, logreg.predict(X2, logreg.model))
rforest.test <- apply_results(test, rforest.predict(X2, rforest.model))

# score
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
  
  results$f1 = round(2 * (results$prec * results$rec) / (results$prec + results$rec), 3)
  results
}

# output for test runs, write file for submission
if(test_run) {
  scores <- data.frame(acc = numeric(), 
                       prec = numeric(), 
                       rec = numeric(), 
                       f1 = numeric())
  scores['gboost',] <- score(gboost.test$Survived, gboost.test$Output)
  scores['logreg',] <- score(logreg.test$Survived, logreg.test$Output)
  scores['rforest',] <- score(rforest.test$Survived, rforest.test$Output)
  arrange(scores, desc(f1))
  print(scores)
}


