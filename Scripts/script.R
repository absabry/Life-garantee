# environnement ####
setwd("C:\\Users\\HP\\Desktop\\ESILVS09\\Data Science\\Apprentissage\\V2")
source('functions.R') # external functions
library(data.table)
require(caret)
require(cluster); library(pvclust); library(fpc)
require(ggplot2) ; library(ggthemes); library(corrplot); library(GGally); library(ggcorrplot);
library(tidyr);library(purrr);#dataviz
library(randomForest); library(xgboost); library(gbm); library(e1071)
library(earth)
#library(devtools)
#install_github("selva86/InformationValue");install_github("riv","tomasgreif/riv");install.packages("Boruta");
library(InformationValue); library(woe);library(Boruta)

trainOriginal = read.csv('train.csv')
validOriginal = read.csv('valid.csv')

# Data Prep #####

trainOriginal = trainOriginal[, !(names(trainOriginal) %in% c("X"))] 
validOriginal = validOriginal[,!(names(validOriginal) %in% c("X"))]
names(trainOriginal)[!(names(trainOriginal) %in% names(validOriginal))] # difference between train and valid : target column
names(validOriginal)[!(names(validOriginal) %in% names(trainOriginal))] # 0 


validOriginal$target=-1
jeux = rbind(trainOriginal,validOriginal)

na.columns = colSums(is.na(jeux))/nrow(jeux) *100 # proprtion of NAs
na.columns = na.columns[na.columns !=0]
print(na.columns) # 17 columns

greatherThan50 = na.columns[na.columns>50]
jeux = jeux[, !(names(jeux) %in% names(greatherThan50))]
isNas = which(is.na(jeux$var_4))
jeux = jeux[-isNas,]
#na.columns = colSums(is.na(jeux))/nrow(jeux) *100 # proprtion of NAs
#na.columns = na.columns[na.columns !=0]
#print(na.columns) # 7 columns ONLY!!


rows = complete.cases(jeux)
length(rows[rows=="TRUE"])*100/nrow(jeux) # 8.14% of complete cases

# Delete Nas from columns
types = sapply(jeux,class)
types.factor = names(types[types == "factor"])
print(types.factor)

for(column in types.factor){
  if(sum(is.na(jeux[,column])) > 0) jeux[is.na(jeux[,column])] <- "Ex.Na"
} 

na.columns = colSums(is.na(jeux))/nrow(jeux) *100 # current proprtion of NAs
na.columns = na.columns[na.columns !=0]
print(na.columns) # 17 columns still, factors dosent have any Na

NAToMedian <- function(x) replace(x, is.na(x), median(x, na.rm = TRUE))
NAToMean <- function(x) replace(x, is.na(x), mean(x, na.rm = TRUE))
jeux.temp = jeux[, !(names(jeux) %in% types.factor)] # exclude factors from the dataframe temporarily
jeux.temp = replace(jeux.temp, TRUE, lapply(jeux.temp , NAToMedian))
jeux = cbind(jeux.temp,jeux[, (names(jeux) %in% types.factor)])
rm(jeux.temp)

na.columns = colSums(is.na(jeux))/nrow(jeux) *100 # current proprtion of NAs
na.columns = na.columns[na.columns !=0] # 0
print(na.columns)

# Dummies variables 
jeuxDummies = cbind(jeux)
ncol(jeuxDummies) # 35
dummies = dummyVars("~.", data=jeuxDummies,fullRank = F) # create dummy variables
jeuxDummies = as.data.frame(predict(dummies,jeuxDummies))
ncol(jeuxDummies) # 123

trainDummies = jeuxDummies[-which(jeuxDummies$target==-1),]; validDummies= jeuxDummies[which(jeuxDummies$target==-1),];
trainOriginal = jeux[-which(jeux$target==-1),]; validOriginal = jeux[which(jeux$target==-1),]; 
validOriginal$target <- NULL; validDummies$target <- NULL; rm(jeux); rm(jeuxDummies)

# DataViz to explore #####

prop.table(table(trainOriginal$target))*100 # proportion des valeurs dans la target

corr = round(cor(trainDummies),1)
ggcorrplot(corr, hc.order = TRUE, 
           type = "lower", 
           lab = TRUE, 
           lab_size = 3, 
           method="circle", 
           colors = c("tomato2", "white", "springgreen3"), 
           title="Correlogram of the database", 
           ggtheme=theme_bw) # correlation 
rm(corr)

ggplot(trainOriginal, aes(x = departement)) +  
  geom_bar(aes(y = ..prop..,fill = factor(target)))+
  ylab("proportions by departement")+
  facet_grid(~sexe) # proportions de departmetns, par sexe


trainOriginal[,c(1:17)] %>%
  gather(-target,key = "var", value = "value") %>% 
  ggplot(aes(x = value, fill=factor(target))) +
  geom_bar(stat = "count")+
  facet_wrap(~ var, scales = "free")+
  theme_bw()# count of every variable

temp = trainOriginal[,c(17:35)]
temp$target = trainOriginal$target
temp %>%
  gather(-target,key = "var", value = "value") %>% 
  ggplot(aes(x = value, fill=factor(target))) +
  geom_bar(stat = "count")+
  facet_wrap(~ var, scales = "free")+
  theme_bw()# count of every variable
rm(temp)

ggplot(trainOriginal, aes(x = age_prospect)) +  
  geom_bar(aes(y = ..prop..,fill = factor(target)))+
  ylab("proportions by departement")

ggplot(aes(trainingOriginal$age_prospecttrainOriginal$id))

# exploration non-supervisée des données #####
mydata <- trainDummies
wss <- (nrow(mydata)-1)*sum(apply(mydata,2,var))
for (i in 2:15) wss[i] <- sum(kmeans(mydata,
                                     centers=i)$withinss)

differences = list()
for(i in 1:13) differences[i] = abs(wss[i+1] - wss[i])
plot(1:15, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares",
     main="Assessing the Optimal Number of Clusters with the Elbow Method",
     pch=20, cex=2)
rm(mydata)
km = kmeans(trainDummies,5)
plotcluster(trainDummies, km$cluster)
km_two = kmeans(trainDummies,2)
plotcluster(trainDummies, km_two$cluster)
out <- cbind(trainDummies, clusterNum = km_two$cluster)
out_two = out[which(out$clusterNum==2),]
out_one = out[which(out$clusterNum==1),]
#>table(out_one$target)
#0     1 
#32879  1172 
#> table(out_two$target)
#0     1 
#32939  1235 
 
km_three = kmeans(trainDummies,3)
plotcluster(trainDummies, km_three$cluster)
out <- cbind(trainDummies, clusterNum = km_three$cluster)
out_three = out[which(out$clusterNum==3),]
out_two = out[which(out$clusterNum==2),]
out_one = out[which(out$clusterNum==1),]
# on ne remarque rien  de sépcial, ils sont bien distribuié ici...
#>table(out_one$target)
#0     1 
#22009   796 
#> table(out_two$target)
#0     1 
#22070   807 
#> table(out_three$target)
#0     1 
#21739   804 



# compute silhouette for K clusters
sample = trainDummies[c(1:15000),]
dis = dist(sample)^2
silhouettes = list()
for(i in 2:7){
  sample.km = kmeans(sample,i)
  sil=silhouette (sample.km$cluster, dis)
  score = mean(sil[,3])
  plot(sil, col=1:i,border = NA)
  abline(v=score, col="red",lty=2)
  silhouettes[[length(silhouettes)+1]] = list(clusters = i, summary = summary(sil), score= score)
}


dataset.pv=pvclust(sampleOriginal) # correlation between variables in a hirearchical clustering
plot(dataset.pv)
pvrect(dataset.pv,alpha = 0.95)
rm(sample)
# pca to get principale components of the base
# On ne peut pas vraiemtn réduire les composents
# les variances sont bien distribué sur tout les variables
ir.pca <- prcomp(~., data = trainDummies,
                 center = TRUE,
                 scale. = TRUE) 



# supervised learning ####
datas = list()
set.seed(123)
splitIndex  = createDataPartition(trainOriginal$target ,p=.75,list=FALSE,times=1)
trainingOriginal = trainOriginal[splitIndex,]
testOriginal = trainOriginal[-splitIndex,]
trainingDummies = trainDummies[splitIndex,]
testDummies = trainDummies[-splitIndex,]
# glm and stepwise #####
glm1<-glm(target~.,family=binomial,data=trainingDummies)
summary(glm1)
glm1.predict= predict(glm1,type="response", newdata=testDummies)
datas[[length(datas)+1]] = list(predict = glm1.predict, color = "green", title="Glm avec toutes les variables")

# keep coluns having p-values less than 0.05
coeffs = summary(glm1)$coefficients
coeffs = data.frame(coeffs)
kept = coeffs[which(coeffs$Pr...z..<0.05),]
names = row.names(kept)
if(names[1] == '(Intercept)') names = names[-1] # delete Intercepet from the variables
na.columns.text = paste(names, collapse = '+')
na.columns.text = paste('target~',na.columns.text)

glm2<-glm(na.columns.text,family=binomial,data=trainingDummies)
summary(glm2)
glm2.predict= predict(glm2,type="response", newdata=testDummies)
datas[[length(datas)+1]]= list(predict = glm2.predict, color = "yellow", title="Toutes les variables, avec p-values <0.05")


# stepwise method
eval<-sample(1:nrow(trainDummies),20000)
training_set.compressed= trainingDummies[eval,]
mod.temp = glm(target~.,family=binomial, data= training_set.compressed)
aa<-step(mod.temp,lower=~1,direction="both")
kept.variables = names(aa$coefficients)
if(kept.variables[1] == '(Intercept)') kept.variables = kept.variables[-1] # delete Intercepet from the variables
na.columns.text = paste(kept.variables, collapse = '+')
na.columns.text = paste('target~',na.columns.text)
 
glm.stepwise<-glm(na.columns.text,family=binomial,data=trainingDummies)
summary(glm.stepwise)
glm.stepwise.predict= predict(glm.stepwise,type="response", newdata=testDummies)
datas[[length(datas)+1]]= list(predict = glm.stepwise.predict, color = "darkorchid4", title="stepwise on 20k sample")

# get intersections between the "important" variables found in the last models
intersect(names,kept.variables)
length(setdiff(names,kept.variables)) # intersection de P<0.05 et stepwise
length(setdiff(kept.variables,names)) # intersection de stepwise et P<0.05
intersect = paste(intersect(kept.variables,names), collapse = '+')
intersect= paste('target~',intersect)

glm.intersect<-glm(intersect,family=binomial,data=trainingDummies)
summary(glm.intersect)
glm.intersect.predict= predict(glm.intersect,type="response", newdata=testDummies)
datas[[length(datas)+1]]= list(predict = glm.intersect.predict, color = "tan3", title="intersect between p-values<0.05 and stepwsie")


# glmnet #####
# trainOriginal$target = ifelse(trainOriginal$target=="yes",1,0)
#train$target = ifelse(train$target==1, "yes","no")
#getModelInfo()$glmnet$type
trainingDummies$target = ifelse(trainingDummies$target==1, "yes","no")
objControl.glmnet = trainControl(method="cv",number=10,returnResamp ='none',summaryFunction = twoClassSummary, classProbs = TRUE)
parametersGrid.glmnet <-   expand.grid(.alpha = seq(0, 0.1, length = 10),
                                       .lambda = seq(0, 0.1, length = 10))
glmnet = train(data.matrix(trainingDummies[,predictorsDummies]),data.matrix(as.factor(trainingDummies$target))
                        ,method="glmnet",trControl=objControl.glmnet,metric="roc", tuneGrid = parametersGrid.glmnet)
summary(objModel.glmnet)
glmnet.predict = predict(object = objModel.glmnet, data.matrix(testDummies[,predictorsDummies]), type='prob')
datas[[length(datas)+1]] = list(predict = glmnet.predict[,2], color = "red", title="glmnet, alpha=0.02 and lamda=0.01")
plot(objModel.glmnet) # stabilité des paramatres




# random forest #####
set.seed(123)
bestmtry <- tuneRF(trainOriginal[,-trainOriginal$target], as.factor(trainOriginal$target),
                   stepFactor=1.5, improve=1e-10,trace=TRUE,plot=TRUE) # 3:7 is the best Ntree found
print(bestmtry)

random.forest <- randomForest(as.factor(target) ~.,
                              data = trainOriginal, ntree=500, replace = TRUE, Mtry=7, do.trace=TRUE)

plot(random.forest)

random.forest.predict= predict(random.forest,type="prob")
datas[[length(datas)+1]] = list(predict =random.forest.predict[,2], color="blue", title="500 trees, Mtry=3 in random forest")


# random forest tuned
trainOriginal$target = ifelse(trainOriginal$target==1,"yes","no")
customRF <- list(type = "Classification", library = "randomForest", loop = NULL)
customRF$parameters <- data.frame(parameter = c("mtry", "ntree"), class = rep("numeric", 2), label = c("mtry", "ntree"))
customRF$grid <- function(x, y, len = NULL, search = "grid") {}
customRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
  randomForest(x, y, mtry = param$mtry, ntree=param$ntree, ...)
}
customRF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
predict(modelFit, newdata)
customRF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
predict(modelFit, newdata, type = "prob")
customRF$sort <- function(x) x[order(x[,1]),]
customRF$levels <- function(x) x$classes

random.forest.objControl = trainControl(method="cv",number=10,returnResamp ='none',summaryFunction = twoClassSummary, classProbs = TRUE)
random.forest.parametersGrid <- expand.grid(.mtry=c(3,7,10), .ntree=c(50, 100, 200))
random.forest.tuned <- train(as.factor(target) ~ ., data=trainOriginal, method=customRF, metric="ROC", trace=TRUE,
                             tuneGrid=random.forest.parametersGrid, trControl=random.forest.objControl,prediction=TRUE)
summary(random.forest.tuned)
plot(random.forest.tuned)
random.forest.tuned.predict = predict(random.forest.tuned$finalModel, type="prob")
datas[[length(datas)+1]] = list(predict = random.forest.tuned.predict[,2], accuracy = random.forest.tuned.accuracy, color = "sienna", title="Tuned random forest")


# gboost #####
#trainingOriginal$target = ifelse(trainingOriginal$target==1, "yes","no")
predictors = names(trainingOriginal)[names(trainingOriginal)!= "target"]
predictorsDummies = names(trainingDummies)[names(trainingDummies)!= "target"]

gmbControl = trainControl(method="cv",number=10,returnResamp ='none', summaryFunction = twoClassSummary, classProbs = TRUE)
gbm = train(trainingOriginal[,predictors], as.factor(trainingOriginal$target), 
                 method = 'gbm', trControl = gmbControl, metric = 'ROC', preProc = c('center', 'scale')
)
summary(gbm)
gbm.predict = predict(object = gbm, testOriginal[,predictors], type='prob') # [[1]] ==> predict 0, [[2]] ==> predict 1
datas[[length(datas)+1]] = list(predict = gbm.predict[,2], color = "blueviolet", title="gboost with caret package")



gbmGrid <-  expand.grid(interaction.depth = c(1,5,10),
                        n.trees = (1:3)*200,
                        shrinkage = c(0.05, 0.1,0.15), 
                        n.minobsinnode=10)

fitControl <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE)

gbmFit <- train(as.factor(target)~., data = trainOriginal, method = "gbm",
                trControl = fitControl,verbose = TRUE,tuneGrid = gbmGrid)
gbmFit.predict = predict(gbmFit, testOriginal[,predictors], type="prob")
datas[[length(datas)+1]] = list(predict = gbmFit.predict[,2], color = "blanchedalmond", title="Tuned gboost")


gbmGrid <-  expand.grid(interaction.depth = 1,
                        n.trees = (3:4)*200,
                        shrinkage = c(0.15,0.2), 
                        n.minobsinnode=10)

gbmFit.second <- train(as.factor(target)~., data = trainOriginal, method = "gbm",
                trControl = fitControl,verbose = TRUE,tuneGrid = gbmGrid)
gbmFit.second.predict = predict(gbmFit.second, testOriginal[,predictors], type="prob")
datas[[length(datas)+1]] = list(predict = gbmFit.second.predict[,2], color = "darksalmon", title="Tuned gboost second time")


# xgboost ######
Gtrain = data.matrix(trainingDummies)
Gtest = data.matrix(testDummies)

x.train <- xgb.DMatrix(Gtrain[,predictorsDummies], label = Gtrain[,"target"])
x.test <- xgb.DMatrix(Gtest[,predictorsDummies], label = Gtest[,"target"])

aucs = list()
for (eta in c(0.115,0.125,0.130)){
  for (colsample in c(0.210,0.225, 0.240)){
    print(paste("eta=",eta,"colsample=",colsample))
    xgb.params <- list(
      "objective"  = "binary:logistic"
      , "eval_metric" = "auc"
      , "eta" = eta
      , "subsample" = 0.9
      , "colsample_bytree" = colsample
      , "max_depth" = 3
    )
    fit.xgb.cv <- xgb.cv(params = xgb.params, data=x.train, nrounds=120,
                         nfold=15, prediction=TRUE, stratified=TRUE,
                         verbose=TRUE, showsd=FALSE)
    aucs[[length(aucs)+1]] = 
    list(eta =eta,colsample =colsample,max_depth =3,
         auc_mean =max(fit.xgb.cv$evaluation_log$test_auc_mean))
  }
}

best_model = aucs[[1]]
for(model in aucs){
  if(max(model$auc_mean)> max(best_model$auc_mean)){
    best_model = model
  }
}

# first best model: eta=0.1, colsample=0.2, max_depth=4
# second best model: eta=0.1, colample=0.2, max_depth=3
# third best model: eta=0.125, colample=0.225, max_depth=3
# fourth best model: eta=0.125, colample=0.21, max_depth=3


# Tune gamma alone
best_gamma = 0
max_test_mean = 0
for(gamma in c(4,6,8,10,12)){
  print(paste("gamma=",gamma))
  xgb.params <- list(
    "objective"  = "binary:logistic"
    , "eval_metric" = "auc"
    , "eta" = best_model$eta
    , "subsample" = 0.9
    , "colsample_bytree" = best_model$colsample
    , "max_depth" = best_model$max_depth
    , "gamma" = gamma
  )
  fit.xgb.cv <- xgb.cv(params = xgb.params, data=x.train, nrounds=120,
                       nfold=10, prediction=TRUE, stratified=TRUE,
                       verbose=TRUE, showsd=FALSE)
  if(max(fit.xgb.cv$evaluation_log$test_auc_mean)> max_test_mean){
    best_gamma = gamma
    max_test_mean = max(fit.xgb.cv$evaluation_log$test_auc_mean)
  }
}
# best_gamma=4

xgb.params <- list(
  "objective"  = "binary:logistic"
  , "eval_metric" = "auc"
  , "eta" = best_model$eta
  , "subsample" = 0.9
  , "colsample_bytree" = best_model$colsample
  , "max_depth" = best_model$max_depth
  , "gamma" = 4
)

fit.xgb.cv <- xgb.cv(params = xgb.params, data=x.train, nrounds=120,
                                          nfold=15, prediction=TRUE, stratified=TRUE,
                                          verbose=TRUE, showsd=FALSE)
summary(fit.xgb.cv)
datas[[length(datas)+1]] = list(predict = fit.xgb.cv$pred, color = "violetred1", title="Xgboost cross-validation only")


xgb <- xgb.train(xgb.params, data=x.train, nrounds = 120)
summary(xgb)
xgb.predict = predict(xgb, newdata=x.test)


# svm ####
set.seed(123)
svmGrid <- expand.grid(sigma= 2^c(-25, -20, -15,-10, -5, 0), C= 2^c(0:5))
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3,
                       classProbs = TRUE, summaryFunction = twoClassSummary)

svm.tuned <- train(as.factor(target) ~., data = trainingOriginal, method = "svmRadial",
                    trControl=trctrl,preProcess = c("center", "scale"),metric = "ROC",
                    tuneGrid = svmGrid,tuneLength = 10)



  
svm = svm(as.factor(target)~., data= trainingOriginal, probability=TRUE)
p = predict(svm, newdata=testOriginal,probability=TRUE)
svm.predict = attr(p, "probabilities")


# curves####
roc_curves(datas,trainOriginal$target,testOriginal$target, Gtrain[,"target"])
lift_curves(datas,trainOriginal$target,testOriginal$target, Gtrain[,"target"])


# importance ####

stepwise = names(aa$coefficients)

evalModel = earth(as.factor(target)~., data=trainOriginal)
ev <- evimp (evalModel)


varImpPlot(random.forest)
varImpPlot(random.forest.tuned$finalModel)


gbmImp <- varImp(gbm, scale = FALSE)
plot(gbmImp)
plot(gbmImp, top = 20)
gbmFitImp <- varImp(gbmFit, scale = FALSE)
plot(gbmFitImp)
plot(gbmFitImp, top = 20) 
gbmFitsecondImp <- varImp(gbmFit.second, scale = FALSE)
plot(gbmFitsecondImp)
plot(gbmFitsecondImp, top = 20)
glmnetImp <- varImp(glmnet, scale = FALSE)
plot(glmnetImp)
plot(glmnetImp, top = 20)


# l'importance des variables factors, informations value
a = sapply(trainOriginal,class)
factors = names(a[a=="factor"])
all_iv <- data.frame(VARS=factors, IV=numeric(length(factors)), STRENGTH=character(length(factors)), stringsAsFactors = F) 
for (factor_var in factors){
  all_iv[all_iv$VARS == factor_var, "IV"] <- InformationValue::IV(X=trainOriginal[, factor_var], Y=trainOriginal$target)
  all_iv[all_iv$VARS == factor_var, "STRENGTH"] <- attr(InformationValue::IV(X=trainOriginal[, factor_var], Y=trainOriginal$target), "howgood")
}
all_iv <- all_iv[order(-all_iv$IV), ] 

imp = iv.mult(trainOriginal,"target",TRUE) # another way to compute informations values
iv.plot.summary(imp)

boruta.train <- Boruta(as.factor(target)~., data = trainOriginal, doTrace = 2)
print(boruta.train)
#Boruta performed 99 iterations in 5.678539 hours.
#31 attributes confirmed important: age_prospect,
#departement, id, sexe, var_10 and 26 more;
#2 attributes confirmed unimportant: var_21, var_9;
#1 tentative attributes left: var_6;
plot(boruta.train, xlab = "", xaxt = "n")
lz<-lapply(1:ncol(boruta.train$ImpHistory),function(i)
boruta.train$ImpHistory[is.finite(boruta.train$ImpHistory[,i]),i])
names(lz) <- colnames(boruta.train$ImpHistory)
Labels <- sort(sapply(lz,median))
axis(side = 1,las=2,labels = names(Labels),
     at = 1:ncol(boruta.train$ImpHistory), cex.axis = 0.7)

getSelectedAttributes(boruta.train, withTentative = F) # important variables
boruta.df <- attStats(final.boruta)