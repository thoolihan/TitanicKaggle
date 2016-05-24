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
                  'numeric', 'integer', 'integer', 'character', 'numeric', 
                  'character', 'character')
train.raw <- read.csv("data/train.csv", 
                      colClasses = column.types,
                      na.strings = c("NA", ""),
                      header = TRUE)
test.raw <- read.csv("data/test.csv", 
                     na.strings = c("NA", ""), 
                     header = TRUE, 
                     colClasses = column.types[-2])

# prepare
titanic.prepare <- function(df, median_age, median_fare) {
  df <- mutate(df,
         Family = as.integer(SibSp + Parch + 1),
         Age = ifelse(is.na(Age), median_age, Age),
         Fare = ifelse(is.na(Fare), median_fare, Fare),
         Embarked = ifelse(is.na(Embarked), 'S', Embarked)) %>%
    dplyr::select(-Name, -Fare, -Cabin, -Ticket)
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
  set.seed(100)
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

build_model <- function(form, learn_method, tune_grid) {
  train(form, 
        data = data.train, 
        method = learn_method,
        metric = "ROC",
        tuneGrid = tune_grid,
        preProcess = c("center", "scale"),
        trControl = cv.ctrl)
}

pls_tg <- data.frame(.ncomp = 3:10)
pls_model <- build_model(Survived ~ .,
                         learn_method = "pls",
                         tune_grid = pls_tg)

rf_tg <- data.frame(.mtry = 2:8)
rf_model <- build_model(Survived ~ .,
                        learn_method = "parRF",
                        tune_grid = rf_tg)

svm_tg <- expand.grid(.cost = 2:4, .gamma = (0:4)/2)
svm_model <- build_model(Survived ~ .,
                         learn_method = "svmLinear2",
                         tune_grid = svm_tg)

lda_model <- train(Survived ~ ., 
                  data = data.train, 
                  method = "lda",
                  metric = "ROC",
                  preProcess = c("center", "scale"),
                  trControl = cv.ctrl)

ba_model <- train(Survived ~ .,
                  data = data.train,
                  method = "bayesglm",
                  metric = "ROC",
                  preProcess = c("center", "scale"),
                  trControl = cv.ctrl)

nn_tg <- expand.grid(.size = 4, .decay = (0:16)/32)
nn_model <- build_model(Survived ~ ., 
                        learn_method = "nnet", 
                        tune_grid = nn_tg)

mlp_tg <- expand.grid(.layer1 = c(4,6), .layer2 = c(4,6), .layer3 = c(4,6))
mlp_model <- build_model(Survived ~ .,
                         learn_method = "mlpML",
                         tune_grid = mlp_tg)

# predict
data.test$pls_output <- predict(pls_model, data.test)
data.test$rf_output <- predict(rf_model, data.test)
data.test$ba_output <- predict(ba_model, data.test)
data.test$svm_output <- predict(svm_model, data.test)
data.test$lda_output <- predict(lda_model, data.test)
data.test$nn_output <- predict(nn_model, data.test)
data.test$mlp_output <- predict(mlp_model, data.test)
vote <- function(x) {
  ifelse(x == 'Survived', 1, 0)
}

data.test <- 
  mutate(data.test, 
         pls = vote(pls_output),
         rf = vote(rf_output),
         ba = vote(ba_output),
         svm = vote(svm_output),
         lda = vote(lda_output),
         nn = vote(nn_output),
         mlp = vote(mlp_output),
         votes = pls + rf + ba + svm + lda + nn + mlp,
         Consensus = factor(ifelse(votes > 3, 'Survived', 'Perished')))

if(test_run) {
  print(plot(pls_model))
  print(plot(rf_model))
  print(plot(svm_model))
  resamp <- resamples(list(pls = pls_model,
                           rf = rf_model,
                           ba = ba_model,
                           svm = svm_model,
                           lda = lda_model,
                           nn = nn_model,
                           mlp = mlp_model))
  print(summary(resamp))
  print(caret::confusionMatrix(data.test$Consensus, data.test$Survived))
} else {
  ts = format(Sys.time(), "%Y.%m.%d.%H.%M.%S")
  write_results <- function(df, name, output_col, subdir = ts) {
    df2 <- mutate(df, Survived = revalue(output_col, c('Perished' = 0, 'Survived' = 1))) %>%
      dplyr::select(PassengerId, Survived)
    dir = paste('output/', subdir, sep = "")
    if(!dir.exists(dir)) {
      dir.create(dir)
    }
    fname <- paste(dir, '/', name, sep = "")
    write.csv(df2, file = fname, row.names = FALSE, quote = FALSE)
    print(paste('wrote', fname))
  }
  write_results(data.test, 'pls.csv', data.test$pls_output)
  write_results(data.test, 'rf.csv', data.test$rf_output)
  write_results(data.test, 'ba.csv', data.test$ba_output)
  write_results(data.test, 'svm.csv', data.test$svm_output)
  write_results(data.test, 'lda.csv', data.test$lda_output)  
  write_results(data.test, 'nn.csv', data.test$nn_output) 
  write_results(data.test, 'nn.csv', data.test$nn_output) 
  write_results(data.test, 'consensus.csv', data.test$Consensus)
}


