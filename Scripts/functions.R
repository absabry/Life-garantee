library(pROC)
library(ROCR)
roc_curves <- function(models, train.target,test.target, Gtrain.target){
  first = TRUE
  legend.text = vector(mode="numeric", length=0)
  legend.color= vector(mode="character", length=0)
  for(model in models){
    # to have random forest with others models in the same plot
    if(length(model$predict) == length(test.target)) {target = test.target; }
    else if(length(model$predict) == length(train.target)) {target = train.target; }
    else{target = Gtrain.target;}
    model.roc = roc(target, model$predict)
    model.auc = as.numeric(model.roc$auc)
    legend.text = c(legend.text,paste(model$title,'AUC=',round(model.auc*100,2),"%"))
    legend.color = c(legend.color,model$color)
    if(first) {
      plot(model.roc, main="ROCs values", col.main="midnightblue", col = model$color)
      first = FALSE
    }
    else plot(model.roc,col=model$color, add=TRUE)
  }
  legend("bottomright",cex=1,lty=c(1,1), lwd=c(2.5,2.5),legend.text,col=legend.color)
}

lift_curves <- function(models, train.target,test.target, Gtrain.target){
  first = TRUE
  legend.text = vector(mode="numeric", length=0)
  legend.color= vector(mode="character", length=0)
  for(model in models){
    if(length(model$predict) == length(test.target)) {target = test.target; }
    else if(length(model$predict) == length(train.target)) {target = train.target; }
    else{target = Gtrain.target;}
    model.pred = prediction(model$predict,target)
    model.perf <- performance(model.pred,"lift","rpp")
    legend.text = c(legend.text,paste(model$title, ",lift of 10% bucket=",TopDecileLift(model$predict,target)))
    legend.color = c(legend.color,model$color)
    if(first) {
      plot(model.perf, main="lift curve", col.main="midnightblue", col = model$color)
      first = FALSE
    }
    else  plot(model.perf,col=model$color, add=TRUE)
  }
  legend.text = c(legend.text,paste("Naive model,lift of 10% bucket=1"))
  legend.color = c(legend.color,"grey")
  abline(h=1, col="grey")
  legend("topright",cex=1,lty=c(1,1), lwd=c(2.5,2.5),legend.text,col=legend.color)
}