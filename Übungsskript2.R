rm(list=ls())
######## get data
x <- read.table('Cameras_Chip20150306.csv', header=TRUE, sep=";", dec=".")

xs <- subset(x, x$Marke == "Panasonic") #[,7:10]

rownames(xs) <- xs$Typ #we use the labels provide in x
#diese beiden Zeilen nur wenn wir oben die Klammer eckige Klammer wegnehmen!
xDF <- xs[,7:13]
#drop(xs$Kameraklasse)
xs
plot(xs)
library(psych)
pairs.panels(xs[,5])
###### get distances

library(dplyr)
xDF <- select(xDF, -Kameraklasse)
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
library(rpart)
library(rpart.plot)

summary(G)

i <- indicesTV(N, fracIS = 0.7)

xs <- as.data.frame(cbind(xs, G)[i$train,]) #convert data into data frame

rpart_fit <- rpart(x$Preis ~ ., data = xDF)
#rpart_fit <- rpart(G[i$train] ~ ., data = xDF[i$train,])
#we fit the G by all remaining variables, save explanatoiry data

rpart_fit <- rpart(G ~ ., data = xDF, minsplit=100) #how many observations at least do we need
prp(rpart_fit, extra=104) #show classification tree

xDF <- as.data.frame(cbind(xs, G))
rpart_pred <- predict(rpart_fit, xDF) #probabilities
Ghat <- levels(G)[apply(rpart_pred,1, which.max)] #estimated group membership
evalTV(Ghat, G, i)
table(evalTV(Ghat, G, i))
library(cluster)
agn_fit <- agnes(xs, method = 'single') #provide dataset xs and metric
agn_fit #anzeigen
plot(as.dendrogram(agn_fit)) #als dendrogramm
#this is bottom-up clustering!

#How can we quickly split a dataset into subgroups

