library(plyr)
library(dplyr)
library(doMC)
registerDoMC(cores = 4)
library(foreach)
library(caret)
library(neuralnet)

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
nn <- neuralnet(Survived ~ ., data = data.train, hidden = 4, lifesign = "minimal",
                linear.output = FALSE, threshold = 0.1)

# predict
data.test$nn_output <- round(compute(nn, data.test))

if(test_run) {
  plot(nn, rep = "best")
  confusionMatrix(data.test$nn_output, data.test$Survived)
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
  write_results(data.test, 'nn.csv', data.test$nn_output)
}


