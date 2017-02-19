setwd("C:\\Users\\Nick\\Dropbox\\Work\\Data Science\\04 - Titanic Survival Rates")

df=read.csv("train.csv", stringsAsFactors = FALSE)

str(df)


## change variables tp factors
#Change survived
df$Survived=factor(df$Survived, 
                   levels=c(0,1),
                   labels =c("died","lived"))

# Explore the gender survival rate
df$Sex=factor(df$Sex) #change the gender variable
table(df$Survived,df$Sex)


#Explore the age survival rate
library(ggplot2)
age_range=cut(df$Age, seq(0,100,10))
qplot(age_range, xlab="Age Range")

df$Age[is.na(df$Age)]=mean(df$Age, na.rm=TRUE) 

ggplot(df, aes(x=Age, fill=Survived))+
  geom_histogram(binwidth = 5,position="fill")+
  ggtitle("Survival percentage amongst the age groups")

table(df$Survived[is.na(df$Age)]) #check percentage of unknown age passengers 

#Explore embark location
df$Embarked[df$Embarked==""]="S"
df$Embarked=factor(df$Embarked, levels=c("S","C","Q"))
table(df$Survived,df$Embarked)

#Passenger Class & fare
table(df$Survived,df$Pclass)

#Missing Fare
df$logfare=log(df$Fare)
df$logfare[df$Fare==0] = mean(log( df$Fare[df$Fare>0])  )


ggplot(df, aes(x=log(Fare), fill=Survived))+
  geom_histogram(binwidth = 0.5,position="fill")+
  ggtitle("Survival percentage agains fare")

#Missing Fare
df$Fare[df$Fare==0] = mean(log( df$Fare[df$Fare>0])  )
df$logfare=log(df$Fare)


table(df$Survived,df$Parch) #parent children
table(df$Survived,df$SibSp) #siblings/spouse


#Mandatory Data Manipulation prior to running the model:
#Age:
df$Age[is.na(df$Age)]=mean(df$Age, na.rm=TRUE) #is 29.69912

#Embarked:
df$Embarked[df$Embarked==""]="S"
df$Embarked=factor(df$Embarked, levels=c("S","C","Q"))

#Missing Fare
df$Fare[df$Fare==0] = mean(log( df$Fare[df$Fare>0])  )
df$logfare=log(df$Fare)

#####Examine Cabin location
#Extract deck information
deck=(strsplit(df$Cabin[df$Cabin!=""],"[^A-Z]")) #Extract characters from cabin entry
df$Deck[df$Cabin!=""]=unlist(lapply(deck, function(x) head(x,1))) #set deck value to

#Replace missing values with UC1, UC2, UC3
df$Deck[df$Pclass==1 & df$Cabin==""]="UC1"
df$Deck[df$Pclass==2 & df$Cabin==""]="UC2"
df$Deck[df$Pclass==3 & df$Cabin==""]="UC3"
df$Deck=as.factor(df$Deck)

side=(strsplit(df$Cabin[df$Cabin!=""],"[A-Z]")) #Extract characters from cabin entry
df$Side[df$Cabin!=""]=as.numeric(lapply(side, function(x) tail(x,1)))
df$SSide[is.na(df$Side)]=0
df$SSide[!is.na(df$Side)]=1
df$Side[is.na(df$Side)]=0
    #df$SSide=df$Side%%2
    #df$SSide=addNA(df$SSide)
    #table(df$Survived,df$SSide)

table(df$Survived,df$Deck)

## Examine the title
df$Title=sapply(df$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
df$Title=trimws(df$Title)

#Rename certain values
df$Title[df$Title %in% c("Mlle","Ms")]="Miss"
df$Title[df$Title %in% c("Mme")]="Mrs"
df$Title[df$Title %in% c("Major","Jonkheer","Don","Capt")]="Sir"
df$Title[df$Title %in% c("the Countess")]="Lady"


##Make Model
##################################3

#Split into training-test data
library(caret)
set.seed(3456)
trainIndex <- createDataPartition(df$Survived, p = .8,list=FALSE)
df_train=df[trainIndex,]
df_test=df[-trainIndex,]


#train_control1<- trainControl(method="cv", 
#                              number=5,
#                              repeats = 10)

#model.t1=train(Survived~Sex+Age+Embarked+logfare+Pclass+SibSp+Deck+Side,
#              preProc=c("center","scale"),
#              data=df_train,
#              method='C5.0',
#              verbose=FALSE)

#trControl=C5.0Control(CF=0.25,minCases = 5)

library(C50)
mc5=C5.0(Survived~Sex+Age+Embarked+logfare+Pclass+SibSp+Parch+Deck+Side+Title+SSide,
         data=df_train,
         trials=10)


newval=predict(mc5, newdata=df_test)
confusionMatrix(newval, df_test$Survived)



df_test$Incorrect=as.factor(newval!=df_test$Survived)


##EXAMINE ERRORS
#See which one values the model predicts incorrect
#newval=predict(mc5, newdata=df_train)
#confusionMatrix(newval, df_train$Survived)
#df_train$Incorrect=as.factor(newval!=df_train$Survived)

mc5_2=C5.0(Incorrect~Sex+Age+Embarked+logfare+Pclass+SibSp+Parch+Deck+Side+SSide,
           data=df_train,
           trials=10,
           cost=matrix(c(0,1,8,0),nrow=2))
#newval=predict(mc5_2, newdata=df_train)
#confusionMatrix(newval, df_train$Incorrect)


pot_wrong=predict(mc5_2, newdata=df_test)
confusionMatrix(pot_wrong, df_test$Incorrect)

#Set the potentially wrong values to different state
pot_wrong=as.logical(pot_wrong)

k=newval[pot_wrong]
k2=k
k2[k=="died"]="lived"
k2[k=="lived"]="died"
newval[pot_wrong]=k2

#Use random forests
rf_model<-train(Survived~Sex+Age+Embarked+logfare+Pclass+SibSp+Parch+Title+Side+SSide+Deck,
                data=df_train,
                method="rf",
                trControl=trainControl(method="cv",number=10),
                prox=TRUE,
                importance=TRUE)
newval=predict(rf_model, newdata=df_test)
confusionMatrix(newval, df_test$Survived)



#Importing testing data
################################################################################
################################################################################
################################################################################
dft=read.csv("test.csv", stringsAsFactors = FALSE)

#Mandatory Data Manipulation prior to running the model:
#Age:
dft$Age[is.na(dft$Age)]=mean(dft$Age, na.rm=TRUE) 

#Embarked:
dft$Embarked[dft$Embarked==""]="S"
dft$Embarked=factor(dft$Embarked, levels=c("S","C","Q"))

#Missing Fare
dft$logfare=log(dft$Fare)
dft$logfare[is.na(dft$Fare)]= mean(log( dft$Fare[dft$Fare>0]), na.rm=TRUE  )

#Deck and Ship Side:
#####Examine Cabin location
#Extract deck information
deck=(strsplit(dft$Cabin[dft$Cabin!=""],"[^A-Z]")) #Extract characters from cabin entry
dft$Deck[dft$Cabin!=""]=unlist(lapply(deck, function(x) head(x,1))) #set deck value to

#Replace missing values with UC1, UC2, UC3
dft$Deck[dft$Pclass==1 & dft$Cabin==""]="UC1"
dft$Deck[dft$Pclass==2 & dft$Cabin==""]="UC2"
dft$Deck[dft$Pclass==3 & dft$Cabin==""]="UC3"
dft$Deck=as.factor(dft$Deck)

side=(strsplit(dft$Cabin[dft$Cabin!=""],"[A-Z]")) #Extract characters from cabin entry
dft$Side[dft$Cabin!=""]=as.numeric(lapply(side, function(x) tail(x,1)))
dft$SSide[is.na(dft$Side)]=0
dft$SSide[!is.na(dft$Side)]=1
dft$Side[is.na(dft$Side)]=0


## Examine the title
dft$Title=sapply(dft$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
dft$Title=trimws(dft$Title)

#Rename certain values
dft$Title[dft$Title %in% c("Mlle","Ms")]="Miss"
dft$Title[dft$Title %in% c("Mme")]="Mrs"
dft$Title[dft$Title %in% c("Major","Jonkheer","Don","Capt")]="Sir"
dft$Title[dft$Title %in% c("the Countess","Dona")]="Lady"




#Train the original on the entire training data
#Still the best is Deck+Side
mc5=C5.0(Survived~Sex+Age+Embarked+logfare+Pclass+SibSp+Parch+Deck+Side+Sside+Title,
         data=df,
         trials=10)
newval=predict(mc5, newdata=df)
confusionMatrix(newval, df$Survived)
df$Incorrect=as.factor(newval!=df$Survived)


#Predict new values
newval=predict(mc5, newdata=dft)


#DO ERROR CORRECTION
mc5_2=C5.0(Incorrect~Sex+Age+Embarked+logfare+Pclass+Deck,
           data=df,
           trials=10,
           cost=matrix(c(0,1,2,0),nrow=2)) #train for conservativeness 
nw=predict(mc5_2, newdata=df)
confusionMatrix(nw, df$Incorrect)


pot_wrong=predict(mc5_2, newdata=dft)

#Set the potentially wrong values to different state
pot_wrong=as.logical(pot_wrong)

k=newval[pot_wrong]
k2=k
k2[k=="died"]="lived"
k2[k=="lived"]="died"
newval[pot_wrong]=k2







dft$Survived=newval
levels(dft$Survived)= c(0,1)

write.csv(dft[c("PassengerId","Survived")],
          file="submission.csv",
          row.names=FALSE,
          quote=FALSE)

