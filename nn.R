library(plyr)
library(dplyr)
library(doMC)
registerDoMC(cores = 4)
library(foreach)
library(caret)
library(neuralnet)

test_run = FALSE

# read
column.types <- c('integer', 'integer', 'integer', 'character', 'factor', 
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
  mutate(df,
         FirstClass = ifelse(Pclass == 1, 1, 0),
         SecondClass = ifelse(Pclass == 2, 1, 0),
         ThirdClass = ifelse(Pclass == 3, 1, 0),
         Sex = ifelse(Sex == "male", 0, 1),
         Family = as.integer(SibSp + Parch + 1),
         Age = ifelse(is.na(Age), median_age, Age),
         Fare = ifelse(is.na(Fare), median_fare, Fare),
         Embarked = ifelse(is.na(Embarked), 'S', Embarked)) %>%
    dplyr::select(-Name, -Fare, -Cabin, -Ticket, -Pclass)
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
nn <- neuralnet(Survived ~ Sex + Age + FirstClass + SecondClass + ThirdClass + Family, 
                data = data.train, 
                hidden = c(6, 4, 2), 
                lifesign = "minimal",
                linear.output = FALSE, 
                threshold = 0.1, 
                rep = 1)

# predict
test <- dplyr::select(data.test, Sex, Age, FirstClass, SecondClass, ThirdClass, Family)
results <- compute(nn, test)
data.test$nn_output <- as.integer(round(results$net.result))

plot(nn, rep = "best")
if(test_run) {
  print(confusionMatrix(data.test$nn_output, data.test$Survived))
} else {
  ts = format(Sys.time(), "%Y.%m.%d.%H.%M.%S")
  df <- mutate(data.test, Survived = nn_output)
  df <-  dplyr::select(df, PassengerId, Survived)
  dir = paste('output/', ts, sep = "")
  if(!dir.exists(dir)){ dir.create(dir) }
  fname <- paste(dir, '/nn.csv', sep = '')
  write.csv(df, file = fname, row.names = FALSE, quote = FALSE)
  print(paste('wrote', fname))
}


