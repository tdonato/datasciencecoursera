library(caret)
library(ggplot2)

# fix for caret
class2ind <- function(cl)
{
  n <- length(cl)
  cl <- as.factor(cl)
  x <- matrix(0, n, length(levels(cl)) )
  x[(1:n) + n*(unclass(cl)-1)] <- 1
  dimnames(x) <- list(names(cl), levels(cl))
  x
}

setwd("~/R")

training <- read.csv("pml-training.csv")
training <- training[,-c(training$user_name,training$x,training$raw_timestamp_part_1,training$raw_timestamp_part_2,training$cvtd_timestamp)]

for (j in 1:length(training)) {
    training[,j] <- as.numeric(training[,j])  
}

testing <- read.csv("pml-testing.csv")

adaGrid <-  expand.grid(mstop = (1:30)*50,
                        nu = 0.1)

nrow(adaGrid)

fitControl <- trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 10,
  ## repeated ten times
  repeats = 1)

set.seed(825)

adaFit <- train(training$classe ~ ., data = training,
                 method = "bstLs", 
                 preProcess = c("center", "scale","knnImpute"),
                 trControl = fitControl, tuneGrid = adaGrid)
                 #verbose = FALSE)
                 # Now specify the exact models 
                 # to evaludate:
                 #tuneGrid = gbmGrid)

trellis.par.set(caretTheme())
qplot(x,y,data=adaFit)