rm(list = ls())

setwd("D:/Dropbox/Bearbeitete Dokumente Uni Basel/Machine Learning 3KP FS2019/MachineLearning")

x <- read.table("Wholesale customers data.csv", sep = ",", header = TRUE)
install.packages("psych")
install.packages("nnet")

########################
####### Support function
########################

indicesTV <- function(n, fracIS = 0.7)
{
  nIS <- round(n*fracIS)
  iShuffle <- sample(n)
  i <- list(train = iShuffle[1:nIS], valid = iShuffle[(nIS+1):n])
  return(i)
}

evalTV <- function(Ghat, G, i) # compute accuracy for training / vaidation
{
  cmT <- table(Ghat[i$train], G[i$train])
  cmV <- table(Ghat[i$valid], G[i$valid])
  accT <- sum(diag(cmT))/sum(cmT)
  accV <- sum(diag(cmV))/sum(cmV)
  return(c(accT, accV))
}


##########################
####### first impressions
##########################

head(x)
summary(x)
plot(x[,c(4,1)])

#------------------
# pre-process
#------------------

# Kategorien benennen
x$Channel <- factor(x$Channel, labels = c("HoReCa", "Retail"))
x$Region <- factor(x$Region, labels = c("Lis", "Opo", "Other"))
names(x)[8] <- "Delicatessen"  # Fehler korrigieren

lnx <- x # logs
lnx[,3:8] <- log10(x[,3:8])

G <- lnx$Channel # categoriy / class variable
xs <- lnx[,3:8] # features (sum-set)

n <- dim(xs)[1] #number of rows

#------------------

summary(x)

boxplot(x$Milk ~ x$Channel)
hist(x$Milk)
plot(x[,3:8], col = x$Channel)

library(psych)
pairs.panels(x[,3:8])

pairs.panels(lnx[,3:8])
detach("package:psych")
plot(lnx[,3:8], col = x$Channel)



####################################
####### linear discriminant analysis
####################################

library(MASS)


i <- indicesTV(n) # divide in training and valid group

lda_fit <- lda(xs, G) # perform linear descriminant analysis
lda_fit <- lda(xs[i$train,], G[i$train]) # perform linear descriminant analysis with just the Training group
options(digits = 2) #rounding of result
plot(lda_fit)
lda_pred <- predict(lda_fit, xs) # predict full dataset

# confusion matrix full sample
cm <- table(lda_pred$class, G) # performance full sample
acc <- sum(diag(cm))/sum(cm) # accurancy
acc

# confusion matrix in sample
cmIS <- table(lda_pred$class[i$train], G[i$train]) # performance in sample
accIS <- sum(diag(cmIS))/sum(cmIS) # accurancy
accIS

# confusion matrix out of sample
cmOS <- table(lda_pred$class[i$valid], G[i$valid]) # performance out of sample
accOS <- sum(diag(cmOS))/sum(cmOS) # accurancy
accOS

# Function for accurancy of training and valid group
evalTV(lda_pred$class, G, i)

# probability table
prop.table(cm)
prop.table(cm,1)
prop.table(cm,2)

boxplot(lda_pred$x~G)
head(lda_pred$posterior) # result from logit
head(lda_pred$class)

# repeated experiments
accEXP <- matrix(NA, 100 , 2)
for (j in 1:dim(accEXP)[1])
{
i <- indicesTV(n)
lda_fit <- lda(xs[i$train,], G[i$train])
lda_pred <- predict(lda_fit, xs)
accEXP[j,] <- evalTV(lda_pred$class, G, i)
}
boxplot(accEXP)
plot(jitter(accEXP,2))


########################################
####### quadrative discriminant analysis
########################################

i <- indicesTV(n)
qda_fit <- qda(xs[i$train,], G[i$train])
qda_pred <- predict(qda_fit, xs)
accQDA <- evalTV(qda_pred$class, G, i)

# repeated experiments
accEXP <- matrix(NA, 100 , 2)
for (j in 1:dim(accEXP)[1])
{
  i <- indicesTV(n)
  qda_fit <- qda(xs[i$train,], G[i$train])
  qda_pred <- predict(qda_fit, xs)
  accEXP[j,] <- evalTV(qda_pred$class, G, i)
}
boxplot(accEXP)
plot(jitter(accEXP,2))


###################################
####### (artificial) neural network
###################################

library("nnet")
xDF <- as.data.frame(cbind(xs, G))

i <- indicesTV(n)
nn_fit <- nnet(G ~ ., data = xDF[i$train, ], size = c(5))
nn_pred <- predict(nn_fit, xDF, type = "class")
accNN <- evalTV(nn_pred, G, i)

# repeated experiments
accNN <- matrix(NA, 100 , 2)
for (j in 1:dim(accEXP)[1])
{
  i <- indicesTV(n)
  nn_fit <- nnet(G ~ ., data = xDF[i$train, ], size = c(5))
  nn_pred <- predict(nn_fit, xDF, type = "class")
  accNN[j,] <- evalTV(nn_pred, G, i)
}
boxplot(accNN)
plot(jitter(accNN,2))



