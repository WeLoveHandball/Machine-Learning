 rm(lists=ls())

setwd("D:/Dropbox/Bearbeitete Dokumente Uni Basel/Machine Learning 3KP FS2019/MachineLearning")
x <- read.table('Wholesale customers data.csv', sep = ',', header = TRUE)

#install.packages('psych')
#kann ausgeklammert werden!!!
#
#first impressions
#

#############
#### support functions
#########

indicesTV <- function(N, fracIS = 0.7) { #training validation aufteilung, 30% training vom set
nIS <- round(N*fracIS)#random nummern generieren, i ist eine liste(bootstrapping)
iShuffle <- sample(N)
#iShuffle(6)
#iShuffle[1:4]
#iShuffle[1:nIS]
#iShuffle[(nIS+1):N]
#iShuffle[-3]
#sample(1:4)

i <- list(train = iShuffle[1:nIS], valid = iShuffle[(nIS+1):N])
return(i)

}

evalTV <- function(Ghat, G, i){
  cmT <- table(Ghat[i$train], G[i$train])
  cmV <- table(Ghat[i$valid], G[i$valid])
  accT <- sum(diag(cmT))/sum(cmT)
  accV <- sum(diag(cmV))/sum(cmV)
  return(c(accT, accV))
}



################
########## first impressions
################

head(x)
summary(x)

#--------------
#pre-process
#--------------

x$Channel <- factor(x$Channel, labels=c('HoReCa', 'Retail'))
x$Region <- factor(x$Region, labels=c('Lis', 'Opo' , 'Other'))
names(x)[8] <- 'Delicatessen'
summary(x)
xl <- x
xl[,3:8] <- log10(x[,3:8])

G <- xl$Channel # category / class variable
xs <- xl[,3:8]  # features (sub-set)

N <- dim(xs)[1] # number of observations in data set

#---------------------

boxplot(x$Milk ~ x$Channel)
hist(x$Milk)
plot(x)
plot(x[,3:8], col=x$Channel)

library(psych)
pairs.panels(x[,3:8])

library(psych)
pairs.panels(xl[,3:8])
detach("package:psych")

plot(xl[,3:8], col=x$Channel)

##########
#### linear discriminant analysis
#########

library(MASS)

i <- indicesTV(N)
lda_fit <- lda(xs[i$train, ], G[i$train]) # perform linear discriminant analysis on training data
#(alle rows von itrain und alle colonnen nach komma angegeben)
help(lda)
options(digits = 2)
lda_fit #was bedeuten diese coeffizienten?
summary(x)
#the new variable Z in the scripts equation is/are the seen coefficients!
plot(lda_fit)
#intrested in the diffrent values
lda_pred <- predict(lda_fit, xs)


evalTV(lda_pred$class, G, i) #
#evalTV(Ghat, G, i)

#repeat experiments 100 times
accExp <- matrix(NA, 100, 2)
for (e in 1:dim(accExp)[1]){
  i <- indicesTV(N)
  lda_fit <- lda(xs[i$train, ], G[i$train]) # perform linear discriminant analysis on training data
  lda_pred <- predict(lda_fit,xs)#perform lda on validation set with our features and lda_fit
  accExp[e,] <- evalTV(lda_pred$class, G, i)
}

accExp[e,]

######################
######### quadratic discriminant analysis
#################

i <- indicesTV(N)
qda_fit <- qda(xs[i$train, ], G[i$train])#fiten unsere features vom training set sowie unsere gruppen vom trainingsset
qda_pred <- predict(qda_fit, xs)
accQDA <- evalTV(qda_pred$class, G, i)
#head(qda_pred$class)
accQDA
table(accQDA) #warum ist hier keine diagonale??

#repeat experiments
accExp <- matrix(NA, 100, 2)
for (e in 1:dim(accExp)[1]){
  i <- indicesTV(N)
  qda_fit <- qda(xs[i$train, ], G[i$train]) # perform quadratic discriminant analysis on training data
  qda_pred <- predict(qda_fit,xs)
  accExp[e,] <- evalTV(qda_pred$class, G, i)
}
plot(qda_fit)
qda_pred
accExp[e,]

######################
######### (artificial) neral networks
#################

library(nnet)
xDF <- as.data.frame(cbind(xs, G))
help(matrix)
help(cbind)
accExp <- matrix(NA, 100, 2)#matrix gefüllt mit NA
for (e in 1:dim(accExp)[1]){
i <- indicesTV(N)
nn_fit <- nnet(G ~ ., data = xDF[i$train, ], size=c(5))
nn_pred <- predict(nn_fit, xDF, type="class")
accExp[e,] <- evalTV(nn_pred, G, i)
}
boxplot(accExp)

accExp[e,]

plot(jitter(accExp,2))#leicht overfitted
#alles mehrmals durchführen => andere Resultate

head(accExp)
boxplot(accExp)

cm <- table(lda_pred$class[i$train], G[i$train])# in sample / training
cm
(accIS <- sum(diag(cm)) / sum(cm)) # accuracy
accIS
cm <- table(lda_pred$class[i$valid], G[i$valid])# out of sample / validation
cm
(accIS <- sum(diag(cm)) / sum(cm)) # accuracy
accIS

head(lda_pred$class)
tail(lda_pred$class)
tail(G)# vergleichen mit Eingabe darüber ob korrekt

(cm <- table(lda_pred$class, G)) #outperform the prior probability 2/3!

diag(cm) #diagonale vom cm, oder table(lda_pred$class, G)
sum(diag(cm)) #aufsumiert zu 403
sum(cm) #insgesamt 440 gibt es und wir haben 403 richtig?

acc <- sum(diag(cm)) / sum(cm) # accuracy
acc #403/440 = 0.92

prop.table(cm) #probabilities = in wahrscheinlichkeiten dargestellt
prop.table(cm,1) #nicht verstanden
prop.table(cm,2)

lda_pred$x
boxplot(lda_pred$x~G)

par(mfrow=c(1,1))
boxplot(lda_pred$x~G)
head(lda_pred$posterior)
head(lda_pred$class)
head(G) #vergleichen ob korrekt mit dem Predictes-Resultat oben

#Gibt unsere Gruppenzuordnung

################
########## support vector machines
################

install.packages("e1071")
library(e1071)

xDF <- as.data.frame(cbind(xs, G)) #feature and target in der klammer

i <- indicesTV(N)
svm_fit <- svm(G ~., data = xDF[i$train,])
svm_fit
svm_pred <- predict(svm_fit, newdata = xDF)
evalTV(svm_pred,G, i) #was bedeutet dieser Output??

plot(svm_fit, data=xDF[i$train, ], Grocery~Frozen) #slice is not good enough

################
########## classification trees
################

install.packages("rpart")
install.packages("rpart.plot")
library(rpart)
library(rpart.plot)

summary(G)

i <- indicesTV(N, fracIS = 0.7)

xDF <- as.data.frame(cbind(xs, G)[i$train,]) #convert data into data frame

rpart_fit <- rpart(G ~ ., data = xDF)
#rpart_fit <- rpart(G[i$train] ~ ., data = xDF[i$train,])
#we fit the G by all remaining variables, save explanatoiry data

rpart_fit <- rpart(G ~ ., data = xDF, minsplit=100) #how many observations at least do we need
prp(rpart_fit, extra=104) #show classification tree

xDF <- as.data.frame(cbind(xs, G))
rpart_pred <- predict(rpart_fit, xDF) #probabilities
Ghat <- levels(G)[apply(rpart_pred,1, which.max)] #estimated group membership
evalTV(Ghat, G, i)
table(evalTV(Ghat, G, i))

################
########## random forest
################

install.packages("randomForest")
library(randomForest)


sample(10)
sample(10,7)
sample(10,7, replace = TRUE)
sample(10,7, replace = TRUE)

#iShuffle <- sample(N,replace=TRUE) #wenn wir ersetzen wollen

rf_fit <- randomForest(xs, G) #diffrent subsamples => boosting, revote
rf_fit
table(rf_fit$predicted, G)

#Versuch adabag
install.packages("adabag")
library(adabag)
boosting

xDF <- as.data.frame(cbind(xs, G))
adabag_fit <- boosting

adabag_pred <- predict(G ~ ., data = xDF)
Ghat <- levels(G)[apply(adabag_pred,1, which.max)]
evalTV(Ghat, G, i)


################
########## naive Bayesian classification
################

install.packages("e1071")
library(e1071)

i <- indicesTV(N, fracIS = 0.7)
nb_fit <- naiveBayes(G[i$train] ~., data = xs[i$train,])
nb_pred <- predict(nb_fit, newdata=xs, laplace=TRUE)

nb_pred

evalTV(nb_pred, G, i)


################
########## nearest neighboors (we are not fitting a model)
################

install.packages("knn")
library(class)
i <- indicesTV(N, fracIS = 0.7)
knn_fit <- knn(xs[i$train, ], xs, G[i$train], 10)
knn_fit
evalTV(knn_fit, G, i)





