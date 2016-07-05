library(ggplot2) 
library(dplyr) 
library(caret)

col.classes <- c('numeric', #PassengerId
                 'numeric', #Survived
                 'numeric', #Pclass
                 'character', #Name
                 'factor', #Sex
                 'numeric', #Age
                 'integer', #SibSp
                 'integer', #Parch
                 'character', #Ticket
                 'numeric', #Fare
                 'character', #Cabin
                 'factor') #Embarked
d.train.raw <- read.csv('./data/train.csv',  
                        colClasses = col.classes,
                        row.names = 1,
                        header = TRUE)

d.submit.raw <- read.csv('./data/test.csv',  
                         colClasses = col.classes[-2],
                         row.names = 1,
                         header = TRUE)

extractTitle <- function(v_names, levels = NA) {
  pattern <- ",\\s*([\\w\\s]+\\.*)"
  matches <- regexpr(pattern, v_names, perl = TRUE)
  res <- regmatches(v_names, matches)
  res <- substring(res, 3, nchar(res) - 1)
  if(sum(is.na(levels)) > 0) {
    res <- factor(res)
  } else {
    res <- factor(res, levels)
  }
}

print('mutating...')
d.train.df <- mutate(d.train.raw,
                     Class = factor(Pclass, levels = as.character(1:3), labels = "class"),
                     Survived = factor(Survived, levels = c(0, 1), labels = c('No', 'Yes')),
                     Title = extractTitle(Name),
                     Deck = factor(ifelse(nchar(Cabin) > 0, substring(Cabin, 1, 1), "X"))) %>%
              select(-Name, -Cabin, -Ticket, -Pclass)

d.submit.df <- mutate(d.submit.raw,
                     Class = factor(Pclass, levels = as.character(1:3), labels = "class"),
                     Title = extractTitle(Name, levels(d.train.df$Title)),
                     Deck = factor(ifelse(nchar(Cabin) > 0, substring(Cabin, 1, 1), "X"))) %>%
  select(-Name, -Cabin, -Ticket, -Pclass)

print('imputing...')
# impute missing data
overall_mean <- mean(d.train.df$Age, na.rm = TRUE)
for(title in levels(d.train.df$Title)){
  index <- d.train.df$Title == title & is.na(d.train.df$Age)
  index2 <- d.submit.df$Title == title & is.na(d.submit.df$Age)
  if(sum(index) > 1) {
    d.train.df[index,]$Age <- mean(d.train.df[d.train.df$Title == title,]$Age, na.rm = TRUE) 
    if(sum(index2) > 0) {
      d.submit.df[index2,]$Age <- mean(d.train.df[d.train.df$Title == title,]$Age, na.rm = TRUE) 
    }
  } else if(sum(index) > 0) {
    d.train.df[index,]$Age <- overall_mean
    if(sum(index2) > 0) {
      d.submit.df[index2,]$Age <- overall_mean
    }
  } else {
    if(sum(index2) > 0) {
      d.submit.df[index2,]$Age <- overall_mean
    }
  }
}
d.submit.df[is.na(d.submit.df$Fare),]$Fare <- 0
d.submit.df[is.na(d.submit.df$Title),]$Title <- 'Mrs'

suppressWarnings(model <- train(Survived ~ ., 
                                data = d.train.df,
                                method = "svmRadialCost",
                                preProcess = c('center', 'scale'),
                                trainControl = trainControl(method = 'cv', repeats = 3, p = .8),
                                tuneGrid = data.frame(.C = 12:20/16)))

print(model)

d.submit.df$predicted <- predict(model, d.submit.df)

write.csv(select(d.submit.df, predicted), 
          file = './output/svm.csv', 
          col.names = c('PassengerId', 'Survived'),
          row.names = TRUE, 
          quote = FALSE)
print('wrote ./output/svm.csv')


