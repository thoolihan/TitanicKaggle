library(plyr)
library(dplyr)
library(doMC)
registerDoMC(cores = 4)
library(foreach)
library(caret)
library(e1071)
library(pROC)

test_run = TRUE

# read
column.types <- c('integer', 'factor', 'factor', 'character', 'factor', 
                  'numeric', 'integer', 'integer', 'character', 'numeric', 'character', 'character')
train.raw <- read.csv("data/train.csv", 
                      colClasses = column.types,
                      na.strings = c("NA", ""),
                      header = TRUE)
test.raw <- read.csv("data/test.csv", header = TRUE, colClasses = column.types[-2])

# prepare
titanic.prepare <- function(df, median_age, median_fare) {
  df <- mutate(df,
         SibSp = SibSp,
         Parch = Parch,
         Family = as.integer(SibSp + Parch + 1),
         Age = ifelse(is.na(Age), median_age, Age),
         Fare = ifelse(is.na(Fare), median_fare, Fare),
         Embarked = ifelse(is.na(Embarked), 'S', Embarked)) %>%
    select(-Name, -Fare, -Cabin, -Ticket)
  df$Pclass <- revalue(df$Pclass, c("1" = "First", "2" = "Second", "3" = "Third"))
  if('Survived' %in% colnames(df)) { 
    df$Survived = revalue(df$Survived, c("0" = "Perished", "1" = "Survived"))
  }
  df
}

# get median values for fields that have NAs
m_age = median(train.raw$Age, na.rm = TRUE)
m_fare = median(train.raw$Fare, na.rm = TRUE)

data.train <- titanic.prepare(train.raw, m_age, m_fare)

# split data
if(test_run) { # split the training set
  set.seed(4)
  train.rows <- createDataPartition(data.train$Survived, p = 0.8, list = FALSE)
  data.test <- data.train[-(train.rows),]
  data.train <- data.train[train.rows,]
} else { # use the test file set
  data.test <- titanic.prepare(test.raw, m_age, m_fare)
  data.test$Survived = NA
}

# train
cv.ctrl <- trainControl(method = "repeatedcv", 
                        repeats = 3,
                        summaryFunction = twoClassSummary,
                        classProbs = TRUE)

pls_tg <- data.frame(.ncomp = 1:23)
pls_model <- train(Survived ~ Sex * Age * Pclass * I(SibSp + Parch), 
             data = data.train, 
             method = "pls",
             metric = "ROC",
             tuneGrid = pls_tg,
             preProcess = c("center", "scale"),
             trControl = cv.ctrl)

rf_tg <- data.frame(.mtry = 2:8)
rf_model <- train(Survived ~ Sex * Age * Pclass * I(SibSp + Parch), 
                   data = data.train, 
                   method = "rf",
                   metric = "ROC",
                   tuneGrid = rf_tg,
                   preProcess = c("center", "scale"),
                   trControl = cv.ctrl)


# predict
data.test$pls_output <- predict(pls_model, data.test)
data.test$rf_output <- predict(rf_model, data.test)

if(test_run) {
  print(plot(pls_model))
  print(plot(rf_model))
  resamp <- resamples(list(pls = pls_model,
                           rf = rf_model))
  print(summary(resamp))
} else {
  ts = format(Sys.time(), "%Y.%m.%d.%H.%M.%S")
  write_results <- function(df, name, output_col, subdir = ts) {
    df <- mutate(df, Survived = revalue(output_col, c('Perished' = 0, 'Survived' = 1))) %>%
      select(PassengerId, Survived)
    dir = paste('output/', subdir, sep = "")
    if(!dir.exists(dir)) {
      dir.create(dir)
    }
    fname <- paste(dir, '/', name, sep = "")
    write.csv(df, file = fname, row.names = FALSE, quote = FALSE)
    print(paste('wrote', fname))
  }
  write_results(data.test, 'pls.csv', data.test$pls_output)
  write_results(data.test, 'rf.csv', data.test$rf_output)
}


