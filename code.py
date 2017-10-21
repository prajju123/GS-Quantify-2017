#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 14:00:34 2017

@author: prajjwal
"""

import os
import pandas as pd
import xgboost as xgb
os.chdir("/home/prajjwal/Downloads/quantify")

train=pd.read_csv("train.csv")
tok_mean=pd.read_csv("tok_mean.csv")
new_train=train.merge(tok_mean)
new_train["mean_rat"]=-1*new_train["mean_rat"]
new_train["pred_used"]=new_train["initialUsedMemory"]+(new_train["mean_rat"]*new_train["cpuTimeTaken"])
new_train["pred_empty"]=new_train["initialFreeMemory"]-(new_train["mean_rat"]*new_train["cpuTimeTaken"])
new_train["total"]=new_train["pred_used"]+new_train["pred_empty"]
new_train["rat1"]=new_train["pred_empty"]/new_train["total"]
new_train["rat2"]=new_train["initialFreeMemory"]/new_train["total"]
new_train.gcRun=new_train.gcRun.astype(int)
columns=[0,1,13,14,15,16,17,18]
X=new_train.iloc[:,columns]
y=new_train.gcRun
clf1 = xgb.XGBClassifier(max_depth=3,n_estimators=60,learning_rate=0.1,scale_pos_weight=11).fit(X,y)

test=pd.read_csv("test.csv")
test["mean_rat"]=0.1
for i in range(1625):
    test["mean_rat"][i]=tok_mean.mean_rat[tok_mean["query token"]==test["query token"][i]]
    
test["pred_used"]=0.001
test["pred_empty"]=0.001
test["total"]=test["pred_used"]+test["pred_empty"]
test["rat1"]=test["pred_empty"]/test["total"]
test["rat2"]=test["initialFreeMemory"]/test["total"]
test["mean_rat"]=-1*test["mean_rat"]
test["pred_used"][0]=test["initialUsedMemory"][0]+(test["mean_rat"][0]*test["cpuTimeTaken"][0])
test["pred_empty"][0]=test["initialFreeMemory"][0]-(test["mean_rat"][0]*test["cpuTimeTaken"][0])
test["total"][0]=test["pred_used"][0]+test["pred_empty"][0]
test["rat1"][0]=test["pred_empty"][0]/test["total"][0]
test["rat2"][0]=test["initialFreeMemory"][0]/test["total"][0]
column=[0,1,5,6,7,8,9,10]

test["gcRun"][0]=clf1.predict(test.iloc[0:1,column])[0]
val=test["pred_used"][0]
val2=test["pred_empty"][0]
for i in range(1,1625):
    print(i)
    test["initialUsedMemory"][i]=test["pred_used"][i-1]
    test["initialFreeMemory"][i]=test["pred_empty"][i-1]
    test["pred_used"][i]=test["initialUsedMemory"][i]+(test["mean_rat"][i]*test["cpuTimeTaken"][i])
    test["pred_empty"][i]=test["initialFreeMemory"][i]-(test["mean_rat"][i]*test["cpuTimeTaken"][i])
    test["total"][i]=test["pred_used"][i]+test["pred_empty"][i]
    test["rat1"][i]=test["pred_empty"][i]/test["total"][i]
    test["rat2"][i]=test["initialFreeMemory"][i]/test["total"][i]
    test["gcRun"][i]=clf1.predict(test.iloc[i:i+1,column])[0]
    if test["gcRun"][i]==1:
        test["pred_used"][i]=val
        test["pred_empty"][i]=val2
       
num=list(range(1,1626))       
ans=pd.DataFrame({'serialNum':num,'initialFreeMemory':test.initialFreeMemory,'gcRun':test.gcRun})        
ans.gcRun[ans.gcRun==1]="TRUE" 
ans.gcRun[ans.gcRun==0]="FALSE" 
ans.to_csv("submit.csv",index=False)      
        
        




    