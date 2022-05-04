rm(list=ls())
######## get data
setwd("D:/Dropbox/Bearbeitete Dokumente Uni Basel/Machine Learning 3KP FS2019/MachineLearning")
x <- read.table('Cameras_Chip20150306.csv', header=TRUE, sep=";", dec=".")

xs <- subset(x, x$Marke == "Canon") #[,7:10]

rownames(xs) <- xs$Typ #we use the labels provide in x
#diese beiden Zeilen nur wenn wir oben die Klammer eckige Klammer wegnehmen!
xs <- xs[,7:10]
xs
plot(xs)
###### get distances

D <- dist(xs) # hier kann man nach xs, die methode wählen, hier: euklydian
options(digits=2)
D #to see all the distances

#library(cluster)
#D <- daisy(x, metr) bei metr kann man auswählen welche methode


####### AGNES (bottom-up)
hc_fit <- hclust(D)
hc_fit #to see how it looks like/information, genauer wenn $-Zeichen
#Dendrogramm
hc_fit <- hclust(D, method = "single")
#we can change the linkage type single average complete
plot(hc_fit, hang = -10)#plot it


#this is based on the linkage type!
#it is the distance matrix

#plot the recdangle for Hclust
rect.hclust(hc_fit, k=2, border = 'blue')
#k stückelt die clusters
rect.hclust(hc_fit, k=3, border = 'red')
#deterministic: this grafical representated solution is not unique!!!
#because we can change the order of letters
#but the height will not change!!!
#we used a greedy algorythm
#Splinter: not looking ahead and maybe looses information
#so do top-down, similar? yes or no =< robust

library(cluster)
agn_fit <- agnes(xs, method = 'single') #provide dataset xs and metric
agn_fit #anzeigen
plot(as.dendrogram(agn_fit)) #als dendrogramm
#this is bottom-up clustering!

#How can we quickly split a dataset into subgroups

######## DIANA (top-down)
diana_fit <- diana(xs)
#we need metric numerical feature
plot(as.dendrogram(diana_fit))

names(x)


##########
#Hierarchical: MONA
########

rm(list=ls())
x <- read.table('MultiMediaSub.csv', header = TRUE, sep=";", dec=".")
#install.packages("tidyverse")
library(tidyverse)
xs <- select(x, starts_with("hobby"))[1:20,]
detach(package:tidyverse)
summary(xs)
(mona_fit <- mona(xs))
plot(mona_fit)



##########
#k-means
########

rm(list=ls())

setwd("D:/Dropbox/Bearbeitete Dokumente Uni Basel/Machine Learning 3KP FS2019/MachineLearning")

x <- read.table('OnlineZeitung.csv', header = TRUE, sep=";", dec = ".")
pairs(x)
dim(x)

k <- 2 #Anzahl clusters
k <- 3
k <- 4

km_fit <- kmeans(x, k)
km_fit #first group isnt interested in economics, but in sports and politics
#which variable is the differs the most, this variable seperates the groups
km_fit$centers

pairs(x,col=km_fit$cluster)#do it a few times, normally they are the same
#sometimes the sportsmen are seperated in 2 groups

#if we have 4 clusters but just 3 groups=> too much, similar clusters are there and are the same cluster

km_fit$betweenss
km_fit$withinss
N <- dim(x)[1]

Fstar <- 0 # Fstatistic, overall best

Fopt <- matrix(0,10,1)
for (k in 2:10){ #try diffrent k
 for (e in 1:30){#we repeat the experiment/fit for given k experiment 1-100
  km_fit <- kmeans(x,k) #we fit
  Fstat <- (km_fit$betweenss/(k-1)) / (km_fit$tot.withinss / (N-k)) #do it in the matrix
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
k <- 3
pam_fit <- pam(x,k)
print(pam_fit)

#or

library(cluster)
k = 3 #6 oder 3 ändern zu k=3
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

plot(x, col=rgb(fc_fit$membership))

#protection down to 2 dimensions: good for visual inspection
mdf <- cmdscale(dist(x))
plot(mdf)
plot(mdf)[,2]
mdf

##########################
##PCA Principal Components Analysis
#######################

eigen(cov(x))

pca <- prcomp(x)
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












