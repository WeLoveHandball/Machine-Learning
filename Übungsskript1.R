rm(lists=ls())

setwd("D:/Dropbox/Bearbeitete Dokumente Uni Basel/Machine Learning 3KP FS2019/MachineLearning")
v <- read.table('fitness.csv', sep = ';', header = TRUE)

head(v)
summary(v)
plot(v)

x$Channel <- factor(x$Channel, labels=c('Spielplatz', 'Klassenzimmer'))
x$Region <- factor(x$Region, labels=c('Land', 'Stadt' , 'Gebirge'))
names(x)[3] <- 'Waldfläche'
names(x)[4] <- 'Strasse'
names(x)[5] <- 'Wohnhäuser'
names(x)[6] <- 'Einkaufsläden'
names(x)[7] <- 'Türen'
names(x)[8] <- 'Bücher'
summary(x)
xl <- x
xl[,3:8] <- log10(x[,3:8])

G <- xl$Channel # category / class variable
xs <- xl[,3:8]  # features (sub-set)

N <- dim(xs)[1] # number of observations in data set


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

install.packages("knn")
library(class)
i <- indicesTV(N, fracIS = 0.7)
knn_fit <- knn(xs[i$train, ], xs, G[i$train], 10)
knn_fit
evalTV(knn_fit, G, i)

rownames(v) <- v$Weight #we use the labels provide in x
#diese beiden Zeilen nur wenn wir oben die Klammer eckige Klammer wegnehmen!
#vs <- v[,1:6]
#vs
#plot(vs)
###### get distances

D <- dist(v) # hier kann man nach xs, die methode wählen, hier: euklydian
options(digits=2)
D #to see all the distances

#library(cluster)
#D <- daisy(x, metr) bei metr kann man auswählen welche methode

####### AGNES (bottom-up)
fitness_fit <- hclust(D)
fitness_fit #to see how it looks like/information, genauer wenn $-Zeichen
#Dendrogramm
fitness_fit <- hclust(D, method = "complete")
#we can change the linkage type single average complete
plot(fitness_fit, hang = -10)#plot it

#plot the recdangle for Hclust
rect.hclust(fitness_fit, k=2, border = 'blue')
#k stückelt die clusters
rect.hclust(fitness_fit, k=3, border = 'red')
#deterministic: this grafical representated solution is not unique!!!
#because we can change the order of letters
#but the height will not change!!!
#we used a greedy algorythm
#Splinter: not looking ahead and maybe looses information
#so do top-down, similar? yes or no =< robust

library(cluster)
fitnessagn_fit <- agnes(v, method = 'single') #provide dataset xs and metric
fitnessagn_fit #anzeigen
plot(as.dendrogram(fitnessagn_fit)) #als dendrogramm
#this is bottom-up clustering!
rect.hclust(fitnessagn_fit, k=3, border = 'red')
#How can we quickly split a dataset into subgroups

######## DIANA (top-down)
fitnessdiana_fit <- diana(v)
#we need metric numerical feature
plot(as.dendrogram(fitnessdiana_fit))
rect.hclust(fitness_fit, k=2, border = 'orange')
rect.hclust(fitness_fit, k=3, border = 'blue')
rect.hclust(fitness_fit, k=4, border = 'red')
rect.hclust(fitness_fit, k=5, border = 'green')
names(v)


##########
#Hierarchical: MONA
########

rm(list=ls())
x <- read.table('MultiMediaSub.csv', header = TRUE, sep=";", dec=".")
#install.packages("tidyverse")
library(tidyverse)
#xs <- x[,9:11]
xs <- select(x, starts_with("Hobby"))[1:20,]
detach(package:tidyverse)
summary(xs)
mona_fit <- mona(xs)
plot(mona_fit)
#warum findet es die Funktion MONA nicht???


##########
#k-means
########

rm(list=ls())

setwd("D:/Dropbox/Bearbeitete Dokumente Uni Basel/Machine Learning 3KP FS2019/MachineLearning")

x <- read.table('Cars_70s_80s.csv', header = TRUE, sep=";", dec = ".")
xs <- x[,2:6]
pairs(xs)
dim(xs)

k <- 2 #Anzahl clusters
k <- 3
k <- 4
k <- 5

fitnesskm_fit <- kmeans(xs, k)
fitnesskm_fit #first group isnt interested in economics, but in sports and politics
#which variable is the differs the most, this variable seperates the groups
fitnesskm_fit$centers

pairs(x,col=fitnesskm_fit$cluster)#do it a few times, normally they are the same
#sometimes the sportsmen are seperated in 2 groups

#if we have 4 clusters but just 3 groups=> too much, similar clusters are there and are the same cluster

fitnesskm_fit$betweenss
fitnesskm_fit$withinss
N <- dim(x)[1]

Fstar <- 0 # Fstatistic, overall best

Fopt <- matrix(0,10,1)
for (k in 2:10){ #try diffrent k
  for (e in 1:30){#we repeat the experiment/fit for given k experiment 1-100
    km_fit <- kmeans(x,k) #we fit
    Fstat <- (fitnesskm_fit$betweenss/(k-1)) / (fitnesskm_fit$tot.withinss / (N-k)) #do it in the matrix
    if (Fstat > Fopt[k]){ # if this experiment has a higher Fstat (for a given k) than previous result
      Fopt[k] <- Fstat
      if (Fstat > Fstar){
        Fstar <- Fstat
        km_fit_star <- km_fit
      }
    }
  }
  
  print(paste(k,Fopt[k]))
}
(k_star <- which.max(Fopt)) # which element of Fopt has highest value?
km_fit_star #show the best overall fit

##########
#pam k-medoids
########

library(cluster)
k <- 5
pam_fit <- pam(x,k)
print(pam_fit)
pam_sil <- silhouette(pam_fit)
plot(pam_sil)

#or

library(cluster)
k = 2 #6 oder 3 ändern zu k=3
pam_fit <- pam(x,k)
print(pam_fit)
pam_sil <- silhouette(pam_fit)
plot(pam_sil)

#####################
####DBSCAN
####################

install.packages("fpc")
library(fpc)

xs = matrix(rnorm(1000),500,2)
oo = apply(xs**2,1,sum)>1
xs[oo,] = xs[oo,]*3
plot(xs)

#####################
####FUZZY
####################

library(cluster)

k <- 3
fc_fit <- fanny(x,k)
fc_fit #the numbers are probabilities and tell us to which cluster they belong/membership
#hard clustering: höchste Probability ist hier der reihe nach dem cluster zugeordnet
head(fc_fit$membership)

plot(x)

#protection down to 2 dimensions: good for visual inspection
mdf <- cmdscale(dist(x))
plot(mdf)


##########################
##PCA Principal Components Analysis
#######################
rm(lists=ls())
eigen(cov(x$psraid,x$qk9a))

pca <- prcomp(x$psraid)
#the last observation is not unrelated to the previous one
pca
summary(pca)#how much of the information can we scan
#the first PC1: we almost collected 2/3 of the information, so we represent 90% of
#variation of the data set with just die first 2 variables
#=< sports hat in der ersten dimension einen grossen wert, die anderen erst in der zweiten dimension
biplot(pca)
#hat jemand sport gerne, wird er nach rechts gedrückt, wenn unter dem average dann nach links
#same results if you do MDS, but rotated (7dimensional)
#how much information can we squeeze into a 2-deminsional plot
#we take a dataset and create a new variable(we need large variation=>better, more information)

#descending order in proportion of variance: sum up in descend order

##########################
##CCA Canonical Correlation Analysis
#######################

rm(lists=ls())
#combination of existing ones
x <- read.table('TeensMobile_Reduced.csv', header = TRUE, sep=";", dec=".")
x <- na.omit(x)
x1 <- x[,9:12]#how often do you talk to...
x2 <- x[,13:17]# topic
#are they correlated?

cc_fit <- cancor(x1,x2)
cc_fit
cc_fit$xcoef
cc_fit$ycoef
cc_fit$cor #what you use your mobilephone for...

#we get dependencies between groups of variables where each individual
#variable gives you not much information, but combined you get much more information
#it depends to who you talk and about what!

#we use these methods if we want to condensate information=>maximaze variance and correlation (sqeeze out)
#gives you the upper limit of information!(upper bounds)

