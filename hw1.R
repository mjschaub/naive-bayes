library(caret)
library(klaR)

#load diabetes data
setwd('D:/CS498/HW1 - naive bayes/')
raw_data<-read.csv('pima-indians-diabetes.txt', header=FALSE)
x_data <- raw_data[-c(9)]
labels <- raw_data[,9]

#1a
training_score<-array(dim=10)
testing_score<-array(dim=10)
for (wi in 1:10){
  
  data_partition <- createDataPartition(y=labels, p=.8, list=FALSE)
  x_test <- x_data[-data_partition,]
  y_test <- labels[-data_partition]
  x_train <- x_data[data_partition,]
  y_train <- labels[data_partition]
  
  
  trposflag<-y_train>0
  ptregs <- x_train[trposflag, ]
  ntregs <- x_train[!trposflag,]
  
  ptrmean<-sapply(ptregs, mean, na.rm=TRUE)
  ntrmean<-sapply(ntregs, mean, na.rm=TRUE)
  ptrsd<-sapply(ptregs, sd, na.rm=TRUE)
  ntrsd<-sapply(ntregs, sd, na.rm=TRUE)
  
  #training set
  ptroffsets<-t(t(x_train)-ptrmean)
  ptrscales<-t(t(ptroffsets)/ptrsd)
  ptrlogs<--(1/2)*rowSums(apply(ptrscales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ptrsd))
  ntroffsets<-t(t(x_train)-ntrmean)
  ntrscales<-t(t(ntroffsets)/ntrsd)
  ntrlogs<--(1/2)*rowSums(apply(ntrscales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ntrsd))
  num_pos_greater_neg_train<-ptrlogs>ntrlogs
  num_correct_train<-num_pos_greater_neg_train==y_train
  training_score[wi]<-sum(num_correct_train)/(sum(num_correct_train)+sum(!num_correct_train))
  
  #test set
  pteoffsets<-t(t(x_test)-ptrmean)
  ptescales<-t(t(pteoffsets)/ptrsd)
  ptelogs<--(1/2)*rowSums(apply(ptescales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ptrsd))
  nteoffsets<-t(t(x_test)-ntrmean)
  ntescales<-t(t(nteoffsets)/ntrsd)
  ntelogs<--(1/2)*rowSums(apply(ntescales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ntrsd))
  num_pos_greater_neg<-ptelogs>ntelogs
  num_correct_test<-num_pos_greater_neg==y_test
  testing_score[wi]<-sum(num_correct_test)/(sum(num_correct_test)+sum(!num_correct_test))
}
accuracy_train <- sum(training_score) / length(training_score)
accuracy_test <- sum(testing_score) / length(testing_score)
accuracy_train
accuracy_test



#1b
#replace 0's with NA
x_data_two <- x_data
for (i in c(3, 4, 6, 8)){
  nan_vals <- x_data[, i]==0
  x_data_two[nan_vals, i]=NA
}


training_score<-array(dim=10)
testing_score<-array(dim=10)
for (wi in 1:10){
  
  data_partition <- createDataPartition(y=labels, p=.8, list=FALSE)
  x_train <- x_data_two[data_partition,]
  y_train <- labels[data_partition]
  x_test <- x_data_two[-data_partition,]
  y_test <- labels[-data_partition]
  
  trposflag<-y_train>0
  ptregs <- x_train[trposflag, ]
  ntregs <- x_train[!trposflag,]
  
  ptrmean<-sapply(ptregs, mean, na.rm=TRUE)
  ntrmean<-sapply(ntregs, mean, na.rm=TRUE)
  ptrsd<-sapply(ptregs, sd, na.rm=TRUE)
  ntrsd<-sapply(ntregs, sd, na.rm=TRUE)
  
  #training stuff
  ptroffsets<-t(t(x_train)-ptrmean)
  ptrscales<-t(t(ptroffsets)/ptrsd)
  ptrlogs<--(1/2)*rowSums(apply(ptrscales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ptrsd))
  ntroffsets<-t(t(x_train)-ntrmean)
  ntrscales<-t(t(ntroffsets)/ntrsd)
  ntrlogs<--(1/2)*rowSums(apply(ntrscales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ntrsd))
  num_pos_greater_neg_train<-ptrlogs>ntrlogs
  num_correct_train<-num_pos_greater_neg_train==y_train
  training_score[wi]<-sum(num_correct_train)/(sum(num_correct_train)+sum(!num_correct_train))
  
  #testing stuff
  pteoffsets<-t(t(x_test)-ptrmean)
  ptescales<-t(t(pteoffsets)/ptrsd)
  ptelogs<--(1/2)*rowSums(apply(ptescales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ptrsd))
  nteoffsets<-t(t(x_test)-ntrmean)
  ntescales<-t(t(nteoffsets)/ntrsd)
  ntelogs<--(1/2)*rowSums(apply(ntescales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ntrsd))
  num_pos_greater_neg<-ptelogs>ntelogs
  num_correct_test<-num_pos_greater_neg==y_test
  testing_score[wi]<-sum(num_correct_test)/(sum(num_correct_test)+sum(!num_correct_test))
}
accuracy_nan_train <- sum(training_score) / length(training_score)
accuracy_nan_test <- sum(testing_score) / length(testing_score)
accuracy_nan_train
accuracy_nan_test





#1c
data_partition <- createDataPartition(y=labels, p=.8, list=FALSE)
x_train <- x_data[data_partition,]
y_train <- labels[data_partition]
x_test <- x_data[-data_partition,]
y_test <- labels[-data_partition]


tr <- trainControl(method='cv' , number=10)
model <- train (x_train , factor(y_train) , 'nb' , trControl=tr)

predictions <- predict(model, newdata=x_test)
cf <- confusionMatrix(data=predictions, y_test)
correct <- length(y_test[y_test == predictions])
wrong <- length(y_test[y_test != predictions])
accuracy <- correct / (correct + wrong)
testing_accuracy <- accuracy

accuracy_cv <- sum(testing_accuracy)/length(testing_accuracy)
cf
accuracy_cv


#1d
data_partition<-createDataPartition(y=labels, p=.8, list=FALSE)
x_train <- x_data[data_partition,]
y_train <- labels[data_partition]
x_test <- x_data[-data_partition,]
y_test <- labels[-data_partition]

#svm stuff
svm <- svmlight(x_train, factor(y_train))
labels <- predict(svm, x_test)
results <- labels$class

correct <- sum(results == y_test)
wrong <- sum(results != y_test)
accuracy_svm <- correct / (correct + wrong)
accuracy_svm



#Problem 2
library(readr)
library(data.table)
setwd('D:/CS498/HW1 - naive bayes/')
raw_train_data_two <- as.data.frame(read.csv("MNIST_train.csv",header=TRUE,check.names=FALSE))


library(caret)
library(klaR)
library(e1071)
y_labels <- (raw_train_data_two$label)
y_labels<-y_labels
y_labels
x_data_mnist <- raw_train_data_two


data_partition <- createDataPartition(y=y_labels, p=.8, list=FALSE)  #[1:30], p=.8, list=FALSE)
x_train <- x_data_mnist[data_partition,]
y_train <- y_labels[data_partition]
x_test <- x_data_mnist[-data_partition,]
y_test <- y_labels[-data_partition]

#train naive bayes model using e1071
model <- naiveBayes(x_train,factor(y_train))
#prediction
predictions <- predict(model, newdata=x_test)
cf <- confusionMatrix(data=predictions, y_test)
correct <- length(y_test[y_test == predictions])
wrong <- length(y_test[y_test != predictions])
accuracy <- correct / (correct + wrong)
testing_accuracy_gaussian_untouched <- accuracy

accuracy_gaussian <- sum(testing_accuracy_gaussian_untouched)/length(testing_accuracy_gaussian_untouched)
cf
accuracy_gaussian


library(quanteda)
library(naivebayes)
thresh = 127
thresh_x_train <- x_train
thresh_x_train[x_train < thresh] <- 0
thresh_x_train[x_train >= thresh] <- 1
thresh_x_test <- x_test
thresh_x_test[x_test < thresh] <- 0
thresh_x_test[x_test >= thresh] <- 1
head(thresh_x_train)
#x_train_dfm <- dfm(as.character(thresh_x_train))
#head(x_train_dfm)
#x_test_dfm <- dfm(as.character(thresh_x_test))
#model_bernoulli <- textmodel_nb(x=x_train_dfm,y=factor(y_train),distribution = c("Bernoulli"))
model_bernoulli <- naive_bayes(x=factor(thresh_x_train, levels=c(0,1)),y=factor(y_train),laplace=1)
model_bernoulli
predictions_b <- predict(model_bernoulli, newdata=thresh_x_test)
cf_b <- confusionMatrix(data=predictions_b, y_test)
correct <- length(y_test[y_test == predictions_b])
wrong <- length(y_test[y_test != predictions_b])
accuracy <- correct / (correct + wrong)
testing_accuracy_bernoulli_untouched <- accuracy

accuracy_bernoulli <- sum(testing_accuracy_bernoulli_untouched)/length(testing_accuracy_bernoulli_untouched)
#accuracy after doing bernoulli naive bayes on MNIST
cf_b
accuracy_bernoulli




rotate_matrix <- function(x) t(apply(x, 2, rev)) #rotates matrix
library(naivebayes)
bounded_m_data_matrix <- matrix(NA,nrow=42000,ncol=401)
#bounded_m_data <- data.frame(matrix(NA, nrow = 42000, ncol = 401))
bounded_m_data_matrix[1:42000,1] <- raw_train_data_two[1:42000,1]
for(x in 1:42000)
{
  curr_m = rotate_matrix(matrix(unlist(raw_train_data_two[x,-1]),nrow = 28,byrow = T))
  prev_matrix <- raw_train_data_two[x,-1]
  curr_m
  thresh = 127
  thresh_m <- curr_m
  thresh_m[curr_m < thresh] <- 0
  thresh_m[curr_m >= thresh] <- 1
  curr_bounded_m <- thresh_m[4:23,4:23]
  curr_bounded_m
  
  new_matrix <- as.vector(curr_bounded_m)
  prev_matrix
  bounded_m_data_matrix[x,-1] <- new_matrix
}
bounded_m_data <- data.frame(bounded_m_data_matrix)


data_partition <- createDataPartition(y=y_labels, p=.8, list=FALSE)  #[1:30], p=.8, list=FALSE)
x_train <- bounded_m_data[data_partition,]
y_train <- y_labels[data_partition]
x_test <- bounded_m_data[-data_partition,]
y_test <- y_labels[-data_partition]

#train naive bayes model using e1071
model <- naiveBayes(x_train,factor(y_train))
#prediction
predictions <- predict(model, newdata=x_test)
cf_bounded <- confusionMatrix(data=predictions, y_test)
correct <- length(y_test[y_test == predictions])
wrong <- length(y_test[y_test != predictions])
accuracy <- correct / (correct + wrong)
testing_accuracy_gaussian_bounded <- accuracy

accuracy_gaussian_bounded <- sum(testing_accuracy_gaussian_bounded)/length(testing_accuracy_gaussian_bounded)
cf_bounded
accuracy_gaussian_bounded



model_bernoulli_bounded <- naive_bayes(x=factor(x_train, levels=c(0,1)),y=factor(y_train),laplace=1)
model_bernoulli_bounded
predictions_b_bounded <- predict(model_bernoulli_bounded, newdata=thresh_x_test)
cf_b_bounded <- confusionMatrix(data=predictions_b, y_test)
correct <- length(y_test[y_test == predictions_b_bounded])
wrong <- length(y_test[y_test != predictions_b_bounded])
accuracy <- correct / (correct + wrong)
testing_accuracy_bernoulli_bounded <- accuracy

accuracy_bernoulli_bounded <- sum(testing_accuracy_bernoulli_bounded)/length(testing_accuracy_bernoulli_bounded)
#accuracy after doing bernoulli naive bayes on MNIST
cf_b_bounded
accuracy_bernoulli_bounded


#decision forest section
library(party)
library(randomForest)
head(raw_train_data_two)
output_forest <- randomForest(label ~ .,data = raw_train_data_two, ntree=30, maxnodes=65536)
output_forest
output_forest_bounded <- randomForest(x=bounded_m_data,y=y_labels,ntree=10,maxnodes=65536)
output_forest_bounded



#maxnodes = 2^depth
#depth 16 - 65536 nodes
#depth 8 - 256 nodes
#depth 4 - 16 nodes









