
train<-read.csv("train.csv")
tr<-train[train$gcRun=="FALSE"]
tr$diff=tr$initialUsedMemory-tr$finalUsedMemory
tr$rat=tr$diff/tr$cpuTimeTaken
tok_mean<-aggregate(tr$rat,by=list(token=query.token),mean)
colnames(tok_mean)=c("query token","mean_rat")
write.csv(tok_mean,"tok_mean.csv",row.names = F)
