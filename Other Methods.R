library(dplyr)
library(ggplot2)
library(h2o)
#install.packages("h2o")
#install.packages("devtools")
library(devtools)
devtools::install_github("tomwenseleers/export")
library(export)

devtools::install_github("gaospecial/ggVennDiagram")
library(ggVennDiagram)


h2o.init(min_mem_size = "800G")


train_vorPandemie           <- miceadds::load.data(file="Mortality Covid AutoMtrain_vorPandemie.RData",type="RData")
test_vorPandemie            <- miceadds::load.data(file="Mortality Covid AutoMtest_vorPandemie.RData",type="RData")
vorPandemie_validierung     <- miceadds::load.data(file="Mortality Covid AutoMvalid_vorPandemie.RData",type="RData")
pandemie                    <- miceadds::load.data(file="Mortality Covid AutoMpandemie.RData",type="RData")

###
# Zusammenfügen
###


train_test <- rbind(train_vorPandemie,test_vorPandemie)



train_test <- train_test %>% filter(year==2019 & month >= 6)

# Split data into partitions
set.seed(3451)
inds <- splitTools::partition(train_test$verstorben, p = c(train = 0.8, test = 0.2))
str(inds)

train_vorPandemie        <- train_test[inds$train, ] 
test_vorPandemie         <- train_test[inds$test, ] 




train_vorPandemie_h20 <- as.h2o(train_vorPandemie)
test_vorPandemie_h20  <- as.h2o(test_vorPandemie)
valid_vorPandemie_h20 <- as.h2o(vorPandemie_validierung)
pandemie_h20          <- as.h2o(pandemie)


# Identify predictors and response
y <- "verstorben"
x <- setdiff(names(train_vorPandemie_h20), y)
#x <- setdiff(x,"gewichtungs_modell")

# For binary classification, response should be a factor
train_vorPandemie_h20[, y] <-  as.factor(train_vorPandemie_h20[, y])
test_vorPandemie_h20[, y]  <- as.factor(test_vorPandemie_h20[, y])
valid_vorPandemie_h20[, y] <- as.factor(valid_vorPandemie_h20[, y])
pandemie_h20[, y]          <- as.factor(pandemie_h20[, y])


# Run AutoML
aml_pandemie_14 <- h2o.automl(x = x, y = y,
                              training_frame = train_vorPandemie_h20,
                              #max_models = 2,
                              seed = 1,
                              validation_frame=test_vorPandemie_h20,
                              blending_frame=test_vorPandemie_h20,
                              balance_classes=TRUE,
                              stopping_metric="AUCPR",
                              sort_metric="AUCPR",
                              stopping_rounds=100,
                              leaderboard_frame = test_vorPandemie_h20,
                              nfolds=0,
                              max_runtime_secs=20*60*60,
                              #exclude_algos="GLM",
                              include_algos = c("GLM", "DeepLearning", "DRF","XGBoost","StackedEnsemble","GBM","DRF"))


save(aml_pandemie_14,file="/ModelleCovid/6monate/aml_pandemie_14.RData")

#aml_pandemie <- miceadds::load.data(file="aml_pandemie_val.RData",type="RData")
#aml_test_pandemie <- miceadds::load.data(file="aml_test_pandemie.RData",type="RData")
#ll <- miceadds::load.data(file="aml_test_pandemie.RData",type="RData")


####
#  Leaderboard Test
####

lb_vorPandemie_h20 <- h2o.get_leaderboard(object = aml_pandemie_14, extra_columns = "ALL")
lb_vorPandemie_h20
save(lb_vorPandemie_h20,file="/ModelleCovid/6monate/lb_vorPandemie_h20.RData")

###
#  Get Models
###

model <- as.vector(aml_pandemie_14@leaderboard$model_id)
model

###
#  Leaderboard in Panemie
###

# Alle Modelle vorhersagen











lb_vorPandemie_h20_df <- as.data.frame(lb_vorPandemie_h20)

lb_vorPandemie_h20_df_Stacked      <- lb_vorPandemie_h20_df %>% filter(algo=="StackedEnsemble")
lb_vorPandemie_h20_df_DeepLearning <- lb_vorPandemie_h20_df %>% filter(algo=="DeepLearning")
lb_vorPandemie_h20_df_DRF          <- lb_vorPandemie_h20_df %>% filter(algo=="DRF")
lb_vorPandemie_h20_df_GBM          <- lb_vorPandemie_h20_df %>% filter(algo=="GBM")
lb_vorPandemie_h20_df_GLM          <- lb_vorPandemie_h20_df %>% filter(algo=="GLM")
lb_vorPandemie_h20_df_XGBoost      <- lb_vorPandemie_h20_df %>% filter(algo=="XGBoost")

openxlsx::write.xlsx(rbind(lb_vorPandemie_h20_df[1,],
                           lb_vorPandemie_h20_df_Stacked[1,],     
                           lb_vorPandemie_h20_df_DeepLearning[1,],
                           lb_vorPandemie_h20_df_DRF[1,],         
                           lb_vorPandemie_h20_df_GBM[1,],         
                           lb_vorPandemie_h20_df_GLM[1,],         
                           lb_vorPandemie_h20_df_XGBoost[1,]     ),"/ModelleCovid/6monate/besteAlgos14.xlsx")





###########################################################################################################################################################################
#################################                                   Leaderboard in pandemie                               #################################################
###########################################################################################################################################################################
pr_ci <- function(x,y){
  nu <- log(x$auc.integral/(1-x$auc.integral))
  tau <- 1/sqrt(y*x$auc.integral*(1-x$auc.integral))
  lower <- exp(nu-1.96*tau)/(1+exp(nu-1.96*tau))
  lower
  upper <- exp(nu+1.96*tau)/(1+exp(nu+1.96*tau))
  upper
  return(paste(round(x$auc.integral,3)," [",round(lower,3)," to ",round(upper,3),"]",sep=""))
  
}
###############################################################################################################################################################################




library(pROC)
library(PRROC)

predictions_vor_pandemie <- data.frame(truth=vorPandemie_validierung$verstorben)
predictions_in_pandemie <- data.frame(truth=pandemie$verstorben)
Leaderboard_Pandemie <- data.frame(modelname=NA,
                                   algo=NA,
                                   AUC_vor_Pandemie=NA,
                                   AUC_mitCI_vor_Pandemie=NA,
                                   PR_vor_Pandemie=NA,
                                   PR_mitCI_vor_Pandemie=NA,
                                   AUC_in_Pandemie=NA,
                                   AUC_mitCI_in_Pandemie=NA,
                                   PR_in_Pandemie=NA,
                                   PR_mitCI_in_Pandemie=NA)

for(i in 1:length(model))
{
  model_loop <-h2o.getModel(model[i])
  
  h2o.saveModel(object = model_loop, path = "/ModelleCovid/6monate/Modellsicherung/", force = TRUE,filename=paste("Mortality_covid_",model[i],sep=""))
  # Predcit vor Pandemie
  
  p_t_vor_pandemie  <- h2o.predict(model_loop, newdata = valid_vorPandemie_h20)
  p_t_in_pandemie   <- h2o.predict(model_loop, newdata = pandemie_h20)
  
  # in Dataframe umwandeln
  
  p_t_vor_pandemie_df  <- as.data.frame(p_t_vor_pandemie  )
  p_t_in_pandemie_df   <- as.data.frame(p_t_in_pandemie   )
  
  
  
  
  predictions_vor_pandemie[,i+1] <- p_t_vor_pandemie_df$p1
  predictions_in_pandemie[,i+1]  <- p_t_in_pandemie_df$p1
  
  # AUC
  aaa_vor_pandemie <- pROC::ci.auc(predictions_vor_pandemie$truth,p_t_vor_pandemie_df$p1)
  aaa_in_pandemie  <- pROC::ci.auc(predictions_in_pandemie$truth,p_t_in_pandemie_df$p1)
  
  # PR
  
  pr_vor_pandemie <- pr.curve(scores.class0 = predictions_vor_pandemie[,i+1][predictions_vor_pandemie$truth==1],  predictions_vor_pandemie[,i+1][predictions_vor_pandemie$truth==0], curve = TRUE)
  pr_in_pandemie  <- pr.curve(scores.class0 = predictions_in_pandemie[,i+1][predictions_in_pandemie$truth==1],  predictions_in_pandemie[,i+1][predictions_in_pandemie$truth==0], curve = TRUE)
  
  
  pr_ci_vor_pandemie <- pr_ci(pr_vor_pandemie,length(predictions_vor_pandemie$truth))
  pr_ci_in_pandemie  <- pr_ci(pr_in_pandemie,length(predictions_in_pandemie$truth))
  
  Leaderboard_Pandemie <- rbind(Leaderboard_Pandemie,
                                c(lb_vorPandemie_h20[i,1][1],lb_vorPandemie_h20[i,10][1],aaa_vor_pandemie[2],paste(sprintf("%.3f",aaa_vor_pandemie[2])," [",sprintf("%.3f",aaa_vor_pandemie[1])," to ",sprintf("%.3f",aaa_vor_pandemie[3]),"]",sep=""),
                                  pr_vor_pandemie$auc.integral,pr_ci_vor_pandemie,
                                  aaa_in_pandemie[2],paste(sprintf("%.3f",aaa_in_pandemie[2])," [",sprintf("%.3f",aaa_in_pandemie[1])," to ",sprintf("%.3f",aaa_in_pandemie[3]),"]",sep=""),
                                  pr_in_pandemie$auc.integral,pr_ci_in_pandemie))
  
  
  
  
  print(paste(i," of ",length(model),sep=""))  
  
}


openxlsx::write.xlsx(Leaderboard_Pandemie,file="/ModelleCovid/6monate/Leaderboard_Pandemie.xlsx")

Leaderboard_Pandemie <- Leaderboard_Pandemie %>% filter(is.na(modelname)==FALSE)

#Leaderboard_Pandemie <- openxlsx::read.xlsx("/ModelleCovid/6monate/Leaderboard_Pandemie.xlsx")
#Leaderboard_Pandemie# Leaderboard_Pandemie
###########################################################################################################################################################################
#################################                             Raussuchen der jeweils besten Algos                         #################################################
###########################################################################################################################################################################
Leaderboard_Pandemie$AUC_vor_Pandemie <- as.numeric(Leaderboard_Pandemie$AUC_vor_Pandemie)
Leaderboard_Pandemie$PR_vor_Pandemie  <- as.numeric(Leaderboard_Pandemie$PR_vor_Pandemie)
Leaderboard_Pandemie$AUC_in_Pandemie  <- as.numeric(Leaderboard_Pandemie$AUC_in_Pandemie)
Leaderboard_Pandemie$PR_in_Pandemie   <- as.numeric(Leaderboard_Pandemie$PR_in_Pandemie)

Leaderboard_Pandemie <- Leaderboard_Pandemie %>% arrange(desc(AUC_vor_Pandemie)) %>% mutate(Reihenfolge.AUC.vor=1:n())
Leaderboard_Pandemie <- Leaderboard_Pandemie %>% arrange(desc(PR_vor_Pandemie)) %>% mutate(Reihenfolge.PR.vor=1:n())
Leaderboard_Pandemie <- Leaderboard_Pandemie %>% arrange(desc(AUC_in_Pandemie)) %>% mutate(Reihenfolge.AUC.in=1:n())
Leaderboard_Pandemie <- Leaderboard_Pandemie %>% arrange(desc(PR_in_Pandemie)) %>% mutate(Reihenfolge.PR.in=1:n())



lb_Stacked      <- Leaderboard_Pandemie %>% filter(algo=="StackedEnsemble")
lb_DeepLearning <- Leaderboard_Pandemie %>% filter(algo=="DeepLearning")
lb_DRF          <- Leaderboard_Pandemie %>% filter(algo=="DRF")
lb_GBM          <- Leaderboard_Pandemie %>% filter(algo=="GBM")
lb_GLM          <- Leaderboard_Pandemie %>% filter(algo=="GLM")
lb_XGBoost      <- Leaderboard_Pandemie %>% filter(algo=="XGBoost")




lb_Stacked      <- lb_Stacked     %>% arrange(desc(PR_vor_Pandemie))
lb_DeepLearning <- lb_DeepLearning%>% arrange(desc(PR_vor_Pandemie))
lb_DRF          <- lb_DRF         %>% arrange(desc(PR_vor_Pandemie))
lb_GBM          <- lb_GBM         %>% arrange(desc(PR_vor_Pandemie))
lb_GLM          <- lb_GLM         %>% arrange(desc(PR_vor_Pandemie))
lb_XGBoost      <- lb_XGBoost     %>% arrange(desc(PR_vor_Pandemie))

beste_vor_pandemie <- rbind(lb_Stacked     [1,],
                            lb_DeepLearning[1,],
                            lb_DRF         [1,],
                            lb_GBM         [1,],
                            lb_GLM         [1,],
                            lb_XGBoost     [1,])

beste_vor_pandemie %>% arrange(desc(PR_vor_Pandemie))


lb_Stacked      <- lb_Stacked     %>% arrange(desc(PR_in_Pandemie))
lb_DeepLearning <- lb_DeepLearning%>% arrange(desc(PR_in_Pandemie))
lb_DRF          <- lb_DRF         %>% arrange(desc(PR_in_Pandemie))
lb_GBM          <- lb_GBM         %>% arrange(desc(PR_in_Pandemie))
lb_GLM          <- lb_GLM         %>% arrange(desc(PR_in_Pandemie))
lb_XGBoost      <- lb_XGBoost     %>% arrange(desc(PR_in_Pandemie))

beste_in_pandemie <- rbind( lb_Stacked     [1,],
                            lb_DeepLearning[1,],
                            lb_DRF         [1,],
                            lb_GBM         [1,],
                            lb_GLM         [1,],
                            lb_XGBoost     [1,])

beste_in_pandemie %>% arrange(desc(PR_in_Pandemie))

openxlsx::write.xlsx(beste_vor_pandemie,"/ModelleCovid/6monate/besteAlgos_vor_pandemie.xlsx")
openxlsx::write.xlsx(beste_in_pandemie,"/ModelleCovid/6monate/besteAlgos_in_pandemie.xlsx")


###########################################################################################################################################################################
#################################                                           Kurven                                        #################################################
###########################################################################################################################################################################
####
#   PR mit besten vor Pandemie
####


best_vor_Pandemie_Stacked      <- lb_Stacked     %>% arrange(desc(PR_vor_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_vor_Pandemie_DeepLearning <- lb_DeepLearning%>% arrange(desc(PR_vor_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_vor_Pandemie_DRF          <- lb_DRF         %>% arrange(desc(PR_vor_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_vor_Pandemie_GBM          <- lb_GBM         %>% arrange(desc(PR_vor_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_vor_Pandemie_GLM          <- lb_GLM         %>% arrange(desc(PR_vor_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_vor_Pandemie_XGBoost      <- lb_XGBoost     %>% arrange(desc(PR_vor_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)


#StackedEnsemble_BestOfFamily_6_AutoML_1_20221205_81006 <- h2o.loadModel("/ModelleCovid/6monate/Modellsicherung/Mortality_covid_StackedEnsemble_BestOfFamily_6_AutoML_1_20221205_81006")


pr_vor_Panemie_Stacked      <- data.frame(pr.curve(scores.class0 = predictions_vor_pandemie[,which(model==best_vor_Pandemie_Stacked     $modelname)+1][predictions_vor_pandemie$truth==1],  predictions_vor_pandemie[,which(model==best_vor_Pandemie_Stacked     $modelname)+1][predictions_vor_pandemie$truth==0], curve = TRUE)$curve)
pr_vor_Panemie_DeepLearning <- data.frame(pr.curve(scores.class0 = predictions_vor_pandemie[,which(model==best_vor_Pandemie_DeepLearning$modelname)+1][predictions_vor_pandemie$truth==1],  predictions_vor_pandemie[,which(model==best_vor_Pandemie_DeepLearning$modelname)+1][predictions_vor_pandemie$truth==0], curve = TRUE)$curve)
pr_vor_Panemie_DRF          <- data.frame(pr.curve(scores.class0 = predictions_vor_pandemie[,which(model==best_vor_Pandemie_DRF         $modelname)+1][predictions_vor_pandemie$truth==1],  predictions_vor_pandemie[,which(model==best_vor_Pandemie_DRF         $modelname)+1][predictions_vor_pandemie$truth==0], curve = TRUE)$curve)
pr_vor_Panemie_GBM          <- data.frame(pr.curve(scores.class0 = predictions_vor_pandemie[,which(model==best_vor_Pandemie_GBM         $modelname)+1][predictions_vor_pandemie$truth==1],  predictions_vor_pandemie[,which(model==best_vor_Pandemie_GBM         $modelname)+1][predictions_vor_pandemie$truth==0], curve = TRUE)$curve)
pr_vor_Panemie_GLM          <- data.frame(pr.curve(scores.class0 = predictions_vor_pandemie[,which(model==best_vor_Pandemie_GLM         $modelname)+1][predictions_vor_pandemie$truth==1],  predictions_vor_pandemie[,which(model==best_vor_Pandemie_GLM         $modelname)+1][predictions_vor_pandemie$truth==0], curve = TRUE)$curve)
pr_vor_Panemie_XGBoost      <- data.frame(pr.curve(scores.class0 = predictions_vor_pandemie[,which(model==best_vor_Pandemie_XGBoost     $modelname)+1][predictions_vor_pandemie$truth==1],  predictions_vor_pandemie[,which(model==best_vor_Pandemie_XGBoost     $modelname)+1][predictions_vor_pandemie$truth==0], curve = TRUE)$curve)



pr_vor_Panemie_Stacked     $dataset <- "Stacked Ensemble"
pr_vor_Panemie_DeepLearning$dataset <- "DeepLearning"
pr_vor_Panemie_DRF         $dataset <- "Deep random forest"
pr_vor_Panemie_GBM         $dataset <- "Gradient boosting machine"
pr_vor_Panemie_GLM         $dataset <- "Generalized linear model"
pr_vor_Panemie_XGBoost     $dataset <- "XGboost"





pr_curve_df <- rbind(pr_vor_Panemie_Stacked     ,
                     pr_vor_Panemie_DeepLearning,
                     pr_vor_Panemie_DRF         ,
                     pr_vor_Panemie_GBM         ,
                     pr_vor_Panemie_GLM         ,
                     pr_vor_Panemie_XGBoost     )

names(pr_curve_df)
names(pr_curve_df) <- c("Recall","Precision","threshold","dataset")

pr_curve_df <- pr_curve_df %>% arrange( dataset,desc(Precision),Recall)


ggplot(data=pr_curve_df ,aes(x=Recall,y=Precision,color=dataset))+
  geom_line(linewidth=2)+
  theme_bw()+
  geom_hline(yintercept = 0.01020408 ,linetype=2)+
  scale_y_continuous(lim=c(0,1),expand = c(0.01, 0.01))+
  scale_x_continuous(expand = c(0.01, 0.01)) +
  theme(axis.title = element_text(size=20),
        axis.text = element_text(size=18,color="black"),
        legend.text = element_text(size=18),
        legend.title = element_text(size=20),
        panel.grid = element_blank(),
        legend.position = c(.8,.8))+
  scale_color_manual("",values=c("#00BF7D","#00B4C5","#0073E6","#B3C7F7","#F57600","#B51963"))
graph2ppt(file="/ModelleCovid/6monate/PowerPointGrafiken/PR beste Algorihmen vor Pandemie.pptx")


###
#  AUC
###

auc_vor_Panemie_Stacked      <- data.frame(sensitivities=pROC::roc(predictions_vor_pandemie$truth,predictions_vor_pandemie[,which(model==best_vor_Pandemie_Stacked     $modelname)+1])$sensitivities,eins_minus_specificities=1-pROC::roc(predictions_vor_pandemie$truth,predictions_vor_pandemie[,which(model==best_vor_Pandemie_Stacked     $modelname)+1])$specificities)
auc_vor_Panemie_DeepLearning <- data.frame(sensitivities=pROC::roc(predictions_vor_pandemie$truth,predictions_vor_pandemie[,which(model==best_vor_Pandemie_DeepLearning$modelname)+1])$sensitivities,eins_minus_specificities=1-pROC::roc(predictions_vor_pandemie$truth,predictions_vor_pandemie[,which(model==best_vor_Pandemie_DeepLearning$modelname)+1])$specificities)
auc_vor_Panemie_DRF          <- data.frame(sensitivities=pROC::roc(predictions_vor_pandemie$truth,predictions_vor_pandemie[,which(model==best_vor_Pandemie_DRF         $modelname)+1])$sensitivities,eins_minus_specificities=1-pROC::roc(predictions_vor_pandemie$truth,predictions_vor_pandemie[,which(model==best_vor_Pandemie_DRF         $modelname)+1])$specificities)
auc_vor_Panemie_GBM          <- data.frame(sensitivities=pROC::roc(predictions_vor_pandemie$truth,predictions_vor_pandemie[,which(model==best_vor_Pandemie_GBM         $modelname)+1])$sensitivities,eins_minus_specificities=1-pROC::roc(predictions_vor_pandemie$truth,predictions_vor_pandemie[,which(model==best_vor_Pandemie_GBM         $modelname)+1])$specificities)
auc_vor_Panemie_GLM          <- data.frame(sensitivities=pROC::roc(predictions_vor_pandemie$truth,predictions_vor_pandemie[,which(model==best_vor_Pandemie_GLM         $modelname)+1])$sensitivities,eins_minus_specificities=1-pROC::roc(predictions_vor_pandemie$truth,predictions_vor_pandemie[,which(model==best_vor_Pandemie_GLM         $modelname)+1])$specificities)
auc_vor_Panemie_XGBoost      <- data.frame(sensitivities=pROC::roc(predictions_vor_pandemie$truth,predictions_vor_pandemie[,which(model==best_vor_Pandemie_XGBoost     $modelname)+1])$sensitivities,eins_minus_specificities=1-pROC::roc(predictions_vor_pandemie$truth,predictions_vor_pandemie[,which(model==best_vor_Pandemie_XGBoost     $modelname)+1])$specificities)


auc_vor_Panemie_Stacked     $dataset <- "Stacked Ensemble"
auc_vor_Panemie_DeepLearning$dataset <- "DeepLearning"
auc_vor_Panemie_DRF         $dataset <- "Deep random forest"
auc_vor_Panemie_GBM         $dataset <- "Gradient boosting machine"
auc_vor_Panemie_GLM         $dataset <- "Generalized linear model"
auc_vor_Panemie_XGBoost     $dataset <- "XGboost"





auc_curve_vor_Pandemie_df <- rbind(auc_vor_Panemie_Stacked     ,
                                   auc_vor_Panemie_DeepLearning,
                                   auc_vor_Panemie_DRF         ,
                                   auc_vor_Panemie_GBM         ,
                                   auc_vor_Panemie_GLM         ,
                                   auc_vor_Panemie_XGBoost     )

names(auc_curve_vor_Pandemie_df)


pr_curve_df <- pr_curve_df %>% arrange( dataset,desc(Precision),Recall)


ggplot(data=auc_curve_vor_Pandemie_df ,aes(x=eins_minus_specificities,y=sensitivities,color=dataset))+
  geom_line(linewidth=2)+
  theme_bw()+
  #geom_hline(yintercept = 0.01020408 ,linetype=2)+
  scale_y_continuous(lim=c(0,1),expand = c(0.01, 0.01),"Sensitivity")+
  scale_x_continuous(expand = c(0.01, 0.01),"1-Specificity") +
  theme(axis.title = element_text(size=20),
        axis.text = element_text(size=18,color="black"),
        legend.text = element_text(size=18),
        legend.title = element_text(size=20),
        panel.grid = element_blank(),
        legend.position = c(.8,.5))+
  scale_color_manual("",values=c("#00BF7D","#00B4C5","#0073E6","#B3C7F7","#F57600","#B51963"))
graph2ppt(file="/ModelleCovid/6monate/PowerPointGrafiken/AUC beste Algorihmen vor Pandemie.pptx")



####
#   PR mit besten in Pandemie
####



best_in_Pandemie_Stacked      <- lb_Stacked     %>% arrange(desc(PR_in_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_in_Pandemie_DeepLearning <- lb_DeepLearning%>% arrange(desc(PR_in_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_in_Pandemie_DRF          <- lb_DRF         %>% arrange(desc(PR_in_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_in_Pandemie_GBM          <- lb_GBM         %>% arrange(desc(PR_in_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_in_Pandemie_GLM          <- lb_GLM         %>% arrange(desc(PR_in_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_in_Pandemie_XGBoost      <- lb_XGBoost     %>% arrange(desc(PR_in_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)





pr_in_Panemie_Stacked      <- data.frame(pr.curve(scores.class0 = predictions_in_pandemie[,which(model==best_in_Pandemie_Stacked     $modelname)+1][predictions_in_pandemie$truth==1],  predictions_in_pandemie[,which(model==best_in_Pandemie_Stacked     $modelname)+1][predictions_in_pandemie$truth==0], curve = TRUE)$curve)
pr_in_Panemie_DeepLearning <- data.frame(pr.curve(scores.class0 = predictions_in_pandemie[,which(model==best_in_Pandemie_DeepLearning$modelname)+1][predictions_in_pandemie$truth==1],  predictions_in_pandemie[,which(model==best_in_Pandemie_DeepLearning$modelname)+1][predictions_in_pandemie$truth==0], curve = TRUE)$curve)
pr_in_Panemie_DRF          <- data.frame(pr.curve(scores.class0 = predictions_in_pandemie[,which(model==best_in_Pandemie_DRF         $modelname)+1][predictions_in_pandemie$truth==1],  predictions_in_pandemie[,which(model==best_in_Pandemie_DRF         $modelname)+1][predictions_in_pandemie$truth==0], curve = TRUE)$curve)
pr_in_Panemie_GBM          <- data.frame(pr.curve(scores.class0 = predictions_in_pandemie[,which(model==best_in_Pandemie_GBM         $modelname)+1][predictions_in_pandemie$truth==1],  predictions_in_pandemie[,which(model==best_in_Pandemie_GBM         $modelname)+1][predictions_in_pandemie$truth==0], curve = TRUE)$curve)
pr_in_Panemie_GLM          <- data.frame(pr.curve(scores.class0 = predictions_in_pandemie[,which(model==best_in_Pandemie_GLM         $modelname)+1][predictions_in_pandemie$truth==1],  predictions_in_pandemie[,which(model==best_in_Pandemie_GLM         $modelname)+1][predictions_in_pandemie$truth==0], curve = TRUE)$curve)
pr_in_Panemie_XGBoost      <- data.frame(pr.curve(scores.class0 = predictions_in_pandemie[,which(model==best_in_Pandemie_XGBoost     $modelname)+1][predictions_in_pandemie$truth==1],  predictions_in_pandemie[,which(model==best_in_Pandemie_XGBoost     $modelname)+1][predictions_in_pandemie$truth==0], curve = TRUE)$curve)



pr_in_Panemie_Stacked     $dataset <- "Stacked Ensemble"
pr_in_Panemie_DeepLearning$dataset <- "DeepLearning"
pr_in_Panemie_DRF         $dataset <- "Deep random forest"
pr_in_Panemie_GBM         $dataset <- "Gradient boosting machine"
pr_in_Panemie_GLM         $dataset <- "Generalized linear model"
pr_in_Panemie_XGBoost     $dataset <- "XGboost"





pr_curve_df <- rbind(pr_in_Panemie_Stacked     ,
                     pr_in_Panemie_DeepLearning,
                     pr_in_Panemie_DRF         ,
                     pr_in_Panemie_GBM         ,
                     pr_in_Panemie_GLM         ,
                     pr_in_Panemie_XGBoost     )

names(pr_curve_df)
names(pr_curve_df) <- c("Recall","Precision","threshold","dataset")

pr_curve_df <- pr_curve_df %>% arrange( dataset,desc(Precision),Recall)


ggplot(data=pr_curve_df ,aes(x=Recall,y=Precision,color=dataset))+
  geom_line(linewidth=2)+
  theme_bw()+
  geom_hline(yintercept = 0.01020408 ,linetype=2)+
  scale_y_continuous(lim=c(0,1),expand = c(0.01, 0.01))+
  scale_x_continuous(expand = c(0.01, 0.01)) +
  theme(axis.title = element_text(size=20),
        axis.text = element_text(size=18,color="black"),
        legend.text = element_text(size=18),
        legend.title = element_text(size=20),
        panel.grid = element_blank(),
        legend.position = c(.8,.8))+
  scale_color_manual("",values=c("#00BF7D","#00B4C5","#0073E6","#B3C7F7","#F57600","#B51963"))
graph2ppt(file="/ModelleCovid/6monate/PowerPointGrafiken/PR beste Algorihmen in Pandemie.pptx")


###
#  AUC
###

auc_in_Panemie_Stacked      <- data.frame(sensitivities=pROC::roc(predictions_in_pandemie$truth,predictions_in_pandemie[,which(model==best_in_Pandemie_Stacked     $modelname)+1])$sensitivities,eins_minus_specificities=1-pROC::roc(predictions_in_pandemie$truth,predictions_in_pandemie[,which(model==best_in_Pandemie_Stacked     $modelname)+1])$specificities)
auc_in_Panemie_DeepLearning <- data.frame(sensitivities=pROC::roc(predictions_in_pandemie$truth,predictions_in_pandemie[,which(model==best_in_Pandemie_DeepLearning$modelname)+1])$sensitivities,eins_minus_specificities=1-pROC::roc(predictions_in_pandemie$truth,predictions_in_pandemie[,which(model==best_in_Pandemie_DeepLearning$modelname)+1])$specificities)
auc_in_Panemie_DRF          <- data.frame(sensitivities=pROC::roc(predictions_in_pandemie$truth,predictions_in_pandemie[,which(model==best_in_Pandemie_DRF         $modelname)+1])$sensitivities,eins_minus_specificities=1-pROC::roc(predictions_in_pandemie$truth,predictions_in_pandemie[,which(model==best_in_Pandemie_DRF         $modelname)+1])$specificities)
auc_in_Panemie_GBM          <- data.frame(sensitivities=pROC::roc(predictions_in_pandemie$truth,predictions_in_pandemie[,which(model==best_in_Pandemie_GBM         $modelname)+1])$sensitivities,eins_minus_specificities=1-pROC::roc(predictions_in_pandemie$truth,predictions_in_pandemie[,which(model==best_in_Pandemie_GBM         $modelname)+1])$specificities)
auc_in_Panemie_GLM          <- data.frame(sensitivities=pROC::roc(predictions_in_pandemie$truth,predictions_in_pandemie[,which(model==best_in_Pandemie_GLM         $modelname)+1])$sensitivities,eins_minus_specificities=1-pROC::roc(predictions_in_pandemie$truth,predictions_in_pandemie[,which(model==best_in_Pandemie_GLM         $modelname)+1])$specificities)
auc_in_Panemie_XGBoost      <- data.frame(sensitivities=pROC::roc(predictions_in_pandemie$truth,predictions_in_pandemie[,which(model==best_in_Pandemie_XGBoost     $modelname)+1])$sensitivities,eins_minus_specificities=1-pROC::roc(predictions_in_pandemie$truth,predictions_in_pandemie[,which(model==best_in_Pandemie_XGBoost     $modelname)+1])$specificities)


auc_in_Panemie_Stacked     $dataset <- "Stacked Ensemble"
auc_in_Panemie_DeepLearning$dataset <- "DeepLearning"
auc_in_Panemie_DRF         $dataset <- "Deep random forest"
auc_in_Panemie_GBM         $dataset <- "Gradient boosting machine"
auc_in_Panemie_GLM         $dataset <- "Generalized linear model"
auc_in_Panemie_XGBoost     $dataset <- "XGboost"





auc_curve_in_Pandemie_df <- rbind(auc_in_Panemie_Stacked     ,
                                  auc_in_Panemie_DeepLearning,
                                  auc_in_Panemie_DRF         ,
                                  auc_in_Panemie_GBM         ,
                                  auc_in_Panemie_GLM         ,
                                  auc_in_Panemie_XGBoost     )

names(auc_curve_in_Pandemie_df)


pr_curve_df <- pr_curve_df %>% arrange( dataset,desc(Precision),Recall)


ggplot(data=auc_curve_in_Pandemie_df ,aes(x=eins_minus_specificities,y=sensitivities,color=dataset))+
  geom_line(linewidth=2)+
  theme_bw()+
  #geom_hline(yintercept = 0.01020408 ,linetype=2)+
  scale_y_continuous(lim=c(0,1),expand = c(0.01, 0.01),"Sensitivity")+
  scale_x_continuous(expand = c(0.01, 0.01),"1-Specificity") +
  theme(axis.title = element_text(size=20),
        axis.text = element_text(size=18,color="black"),
        legend.text = element_text(size=18),
        legend.title = element_text(size=20),
        panel.grid = element_blank(),
        legend.position = c(.8,.5))+
  scale_color_manual("",values=c("#00BF7D","#00B4C5","#0073E6","#B3C7F7","#F57600","#B51963"))
graph2ppt(file="/ModelleCovid/6monate/PowerPointGrafiken/AUC beste Algorihmen in Pandemie.pptx")



###########################################################################################################################################################################
#################################                                         Importance                                      #################################################
###########################################################################################################################################################################

best_vor_Pandemie_Stacked      <- lb_Stacked     %>% arrange(desc(PR_vor_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_vor_Pandemie_DeepLearning <- lb_DeepLearning%>% arrange(desc(PR_vor_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_vor_Pandemie_DRF          <- lb_DRF         %>% arrange(desc(PR_vor_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_vor_Pandemie_GBM          <- lb_GBM         %>% arrange(desc(PR_vor_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_vor_Pandemie_GLM          <- lb_GLM         %>% arrange(desc(PR_vor_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_vor_Pandemie_XGBoost      <- lb_XGBoost     %>% arrange(desc(PR_vor_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)



best_stacked      <- h2o.loadModel(paste("/ModelleCovid/6monate/Modellsicherung/Mortality_covid_",best_vor_Pandemie_Stacked$modelname,sep=""))
best_DeepLearning <- h2o.loadModel(paste("/ModelleCovid/6monate/Modellsicherung/Mortality_covid_",best_vor_Pandemie_DeepLearning$modelname,sep=""))
best_DR           <- h2o.loadModel(paste("/ModelleCovid/6monate/Modellsicherung/Mortality_covid_",best_vor_Pandemie_DRF         $modelname,sep=""))
best_GBM          <- h2o.loadModel(paste("/ModelleCovid/6monate/Modellsicherung/Mortality_covid_",best_vor_Pandemie_GBM         $modelname,sep=""))
best_GLM          <- h2o.loadModel(paste("/ModelleCovid/6monate/Modellsicherung/Mortality_covid_",best_vor_Pandemie_GLM         $modelname,sep=""))
best_XGBoost      <- h2o.loadModel(paste("/ModelleCovid/6monate/Modellsicherung/Mortality_covid_",best_vor_Pandemie_XGBoost     $modelname,sep=""))





vi_vor_Pandemie_Stacked     <- h2o.permutation_importance(h2o.getModel(best_vor_Pandemie_Stacked$modelname), newdata = train_vorPandemie_h20)
vi_vor_Pandemie_DeepLearning<- h2o.varimp(h2o.getModel(best_vor_Pandemie_DeepLearning$modelname))
vi_vor_Pandemie_DRF         <- h2o.varimp(h2o.getModel(best_vor_Pandemie_DRF         $modelname))
vi_vor_Pandemie_GBM         <- h2o.varimp(h2o.getModel(best_vor_Pandemie_GBM         $modelname))
vi_vor_Pandemie_GLM         <- h2o.varimp(h2o.getModel(best_vor_Pandemie_GLM         $modelname))
vi_vor_Pandemie_XGBoost     <- h2o.varimp(h2o.getModel(best_vor_Pandemie_XGBoost     $modelname))


write.csv2(vi_vor_Pandemie_Stacked     ,file="/ModelleCovid/6monate/vi_vor_Pandemie_Stacked.csv",sep=",",dec=";")
write.csv2(vi_vor_Pandemie_DeepLearning,file="/ModelleCovid/6monate/vi_vor_Pandemie_DeepLearning.csv",sep=",",dec=";")
write.csv2(vi_vor_Pandemie_DRF         ,file="/ModelleCovid/6monate/vi_vor_Pandemie_DRF.csv",sep=",",dec=";")
write.csv2(vi_vor_Pandemie_GBM         ,file="/ModelleCovid/6monate/vi_vor_Pandemie_GBM.csv",sep=",",dec=";")
write.csv2(vi_vor_Pandemie_GLM         ,file="/ModelleCovid/6monate/vi_vor_Pandemie_GLM.csv",sep=",",dec=";")
write.csv2(vi_vor_Pandemie_XGBoost     ,file="/ModelleCovid/6monate/vi_vor_Pandemie_XGBoost.csv",sep=",",dec=";")

# vi_vor_Pandemie_Stacked     <- read.csv(file="/ModelleCovid/6monate/vi_vor_Pandemie_Stacked.csv",sep=";",dec=",")
# vi_vor_Pandemie_DeepLearning<- read.csv(file="/ModelleCovid/6monate/vi_vor_Pandemie_DeepLearning.csv",sep=";",dec=",")
# vi_vor_Pandemie_DRF         <- read.csv(file="/ModelleCovid/6monate/vi_vor_Pandemie_DRF.csv",sep=";",dec=",")
# vi_vor_Pandemie_GBM         <- read.csv(file="/ModelleCovid/6monate/vi_vor_Pandemie_GBM.csv",sep=";",dec=",")
# vi_vor_Pandemie_GLM         <- read.csv(file="/ModelleCovid/6monate/vi_vor_Pandemie_GLM.csv",sep=";",dec=",")
# vi_vor_Pandemie_XGBoost     <- read.csv(file="/ModelleCovid/6monate/vi_vor_Pandemie_XGBoost.csv",sep=";",dec=",")


vi_vor_pandemie <- list(Stacked_Ensemble=vi_vor_Pandemie_Stacked $Variable[1:100]    ,
                        DeepLearning=vi_vor_Pandemie_DeepLearning$variable[1:100],
                        DRF=vi_vor_Pandemie_DRF$variable[1:100]        ,
                        GBM=vi_vor_Pandemie_GBM$variable[1:100]        ,
                        GLM=vi_vor_Pandemie_GLM$variable[1:100]         ,
                        XGboost=vi_vor_Pandemie_XGBoost$variable[1:100]     )


vd <- ggVennDiagram(vi_vor_pandemie)
ggVennDiagram(vi_vor_pandemie,show_intersect = F,label_color = "white",label="count")
graph2ppt(file="/ModelleCovid/6monate/PowerPointGrafiken/Venn_Pandemie.pptx")



###
# In Pandemie
###











best_in_Pandemie_Stacked      <- Leaderboard_Pandemie %>% filter(algo=="StackedEnsemble")%>% arrange(desc(PR_in_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_in_Pandemie_DeepLearning <- Leaderboard_Pandemie %>% filter(algo=="DeepLearning")   %>% arrange(desc(PR_in_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_in_Pandemie_DRF          <- Leaderboard_Pandemie %>% filter(algo=="DRF")            %>% arrange(desc(PR_in_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_in_Pandemie_GBM          <- Leaderboard_Pandemie %>% filter(algo=="GBM")            %>% arrange(desc(PR_in_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_in_Pandemie_GLM          <- Leaderboard_Pandemie %>% filter(algo=="GLM")            %>% arrange(desc(PR_in_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_in_Pandemie_XGBoost      <- Leaderboard_Pandemie %>% filter(algo=="XGBoost")        %>% arrange(desc(PR_in_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)



best_stacked      <- h2o.loadModel(paste("/ModelleCovid/6monate/Modellsicherung/Mortality_covid_",best_in_Pandemie_Stacked$modelname,sep=""))
best_DeepLearning <- h2o.loadModel(paste("/ModelleCovid/6monate/Modellsicherung/Mortality_covid_",best_in_Pandemie_DeepLearning$modelname,sep=""))
best_DR           <- h2o.loadModel(paste("/ModelleCovid/6monate/Modellsicherung/Mortality_covid_",best_in_Pandemie_DRF         $modelname,sep=""))
best_GBM          <- h2o.loadModel(paste("/ModelleCovid/6monate/Modellsicherung/Mortality_covid_",best_in_Pandemie_GBM         $modelname,sep=""))
best_GLM          <- h2o.loadModel(paste("/ModelleCovid/6monate/Modellsicherung/Mortality_covid_",best_in_Pandemie_GLM         $modelname,sep=""))
best_XGBoost      <- h2o.loadModel(paste("/ModelleCovid/6monate/Modellsicherung/Mortality_covid_",best_in_Pandemie_XGBoost     $modelname,sep=""))





vi_in_Pandemie_Stacked     <- h2o.permutation_importance(h2o.getModel(best_in_Pandemie_Stacked$modelname), newdata = pandemie_h20)
vi_in_Pandemie_DeepLearning<- h2o.varimp(h2o.getModel(best_in_Pandemie_DeepLearning$modelname))
vi_in_Pandemie_DRF         <- h2o.varimp(h2o.getModel(best_in_Pandemie_DRF         $modelname))
vi_in_Pandemie_GBM         <- h2o.varimp(h2o.getModel(best_in_Pandemie_GBM         $modelname))
vi_in_Pandemie_GLM         <- h2o.varimp(h2o.getModel(best_in_Pandemie_GLM         $modelname))
vi_in_Pandemie_XGBoost     <- h2o.varimp(h2o.getModel(best_in_Pandemie_XGBoost     $modelname))


write.csv2(vi_in_Pandemie_Stacked     ,file="/ModelleCovid/6monate/vi_in_Pandemie_Stacked.csv",sep=",",dec=";")
write.csv2(vi_in_Pandemie_DeepLearning,file="/ModelleCovid/6monate/vi_in_Pandemie_DeepLearning.csv",sep=",",dec=";")
write.csv2(vi_in_Pandemie_DRF         ,file="/ModelleCovid/6monate/vi_in_Pandemie_DRF.csv",sep=",",dec=";")
write.csv2(vi_in_Pandemie_GBM         ,file="/ModelleCovid/6monate/vi_in_Pandemie_GBM.csv",sep=",",dec=";")
write.csv2(vi_in_Pandemie_GLM         ,file="/ModelleCovid/6monate/vi_in_Pandemie_GLM.csv",sep=",",dec=";")
write.csv2(vi_in_Pandemie_XGBoost     ,file="/ModelleCovid/6monate/vi_in_Pandemie_XGBoost.csv",sep=",",dec=";")



####
# Extrahieren der Übereinstimmungen


Intersect <- function (x) {  
  # Multiple set version of intersect
  # x is a list
  if (length(x) == 1) {
    unlist(x)
  } else if (length(x) == 2) {
    intersect(x[[1]], x[[2]])
  } else if (length(x) > 2){
    intersect(x[[1]], Intersect(x[-1]))
  }
}

Union <- function (x) {  
  # Multiple set version of union
  # x is a list
  if (length(x) == 1) {
    unlist(x)
  } else if (length(x) == 2) {
    union(x[[1]], x[[2]])
  } else if (length(x) > 2) {
    union(x[[1]], Union(x[-1]))
  }
}

Setdiff <- function (x, y) {
  # Remove the union of the y's from the common x's. 
  # x and y are lists of characters.
  xx <- Intersect(x)
  yy <- Union(y)
  setdiff(xx, yy)
}

combs <- 
  unlist(lapply(1:length(vi_vor_pandemie), 
                function(j) combn(names(vi_vor_pandemie), j, simplify = FALSE)),
         recursive = FALSE)
names(combs) <- sapply(combs, function(i) paste0(i, collapse = ""))


elements <- 
  lapply(combs, function(i) Setdiff(vi_vor_pandemie[i], vi_vor_pandemie[setdiff(names(vi_vor_pandemie), i)]))

n.elements <- sapply(elements, length)
print(n.elements)

cbind.fill<-function(...){
  nm <- list(...) 
  nm<-lapply(nm, as.matrix)
  n <- max(sapply(nm, nrow)) 
  do.call(cbind, lapply(nm, function (x) 
    rbind(x, matrix(, n-nrow(x), ncol(x))))) 
}
df <-  as.data.frame(do.call(cbind.fill, elements))

names(df) <- names(elements)

openxlsx::write.xlsx(df,"/ModelleCovid/6monate/Übereinstimmungen Venn Diagramm.xlsx")


str(vd)

graph2ppt(file="/ModelleCovid/6monate/VennDiagramm Top100.pptx")

openxlsx::write.xlsx(Reduce(intersect, vi_vor_pandemie),"/ModelleCovid/6monate/Intersect Variablen aller Modelle.xlsx")


###########################################################################################################################################################################
#################################                                       Plot Veränderung                                  #################################################
###########################################################################################################################################################################

library(dplyr)
library(ggplot2)
library(ggridges)
library(export)


head(Leaderboard_Pandemie)


Leaderboard_Pandemie_vor <- Leaderboard_Pandemie %>% select(modelname,algo,AUC_vor_Pandemie,PR_vor_Pandemie) %>% rename(AUC=AUC_vor_Pandemie,PR=PR_vor_Pandemie)
Leaderboard_Pandemie_in  <- Leaderboard_Pandemie %>% select(modelname,algo,AUC_in_Pandemie,PR_in_Pandemie)   %>% rename(AUC=AUC_in_Pandemie,PR=PR_in_Pandemie)

Leaderboard_Pandemie_vor$Zeitpunkt <- "vor"
Leaderboard_Pandemie_in $Zeitpunkt <- "in"


Leaderboard_Pandemie_long <- rbind(Leaderboard_Pandemie_vor,
                                   Leaderboard_Pandemie_in )
Leaderboard_Pandemie_long <- na.omit(Leaderboard_Pandemie_long)

Leaderboard_Pandemie_long$Zeitpunkt <- factor(Leaderboard_Pandemie_long$Zeitpunkt,levels  =c("vor","in"))
Leaderboard_Pandemie_long$AUC <- as.numeric(as.character(Leaderboard_Pandemie_long$AUC))
Leaderboard_Pandemie_long$PR <- as.numeric(as.character(Leaderboard_Pandemie_long$PR))



ggplot(data=Leaderboard_Pandemie_long,aes(x=Zeitpunkt,y=AUC,color=modelname))+geom_line(aes(group=modelname))+geom_point(size=4,shape=21,stroke=2,fill="white") + theme_bw() + theme(legend.position = "none",panel.grid = element_blank(),axis.text = element_text(size=48,color="black"),axis.title=element_text(size=54))+scale_x_discrete("",labels=c("Three month before first wave","In first wave"))+scale_y_continuous("AUROC")
ggsave("/ModelleCovid/6monate/AUROC Veränderung.jpg")
graph2ppt(file="/ModelleCovid/6monate/AUROC Veränderung.pptx",margins = c(top = 0, right = 0, bottom = 0, left = 0),scaling = 200,aspectr=1)

ggplot(data=Leaderboard_Pandemie_long,aes(x=Zeitpunkt,y=PR,color=modelname)) +geom_line(aes(group=modelname))+geom_point(size=4,shape=21,stroke=2,fill="white") + theme_bw() + theme(legend.position = "none",panel.grid = element_blank(),axis.text = element_text(size=16,color="black"),axis.title=element_text(size=18))+scale_x_discrete("",labels=c("Three month before first wave","In first wave"))+scale_y_continuous("AUPR")
ggsave("/ModelleCovid/6monate/AUPR Veränderung.jpg")
graph2ppt(file="/ModelleCovid/6monate/AUPR Veränderung.pptx",margins = c(top = 0, right = 0, bottom = 0, left = 0),scaling = 200,aspectr=1)


ggplot(data=Leaderboard_Pandemie_long,aes(x=Zeitpunkt,y=AUC,color=modelname))+geom_line(aes(group=modelname))+geom_point(size=4,shape=21,stroke=2,fill="white") + theme_bw() +facet_grid(~algo) + theme(legend.position = "none",panel.grid = element_blank(),axis.text = element_text(size=16,color="black"),axis.title=element_text(size=18),strip.background = element_rect(fill="white"),strip.text = element_text(size=18))+scale_x_discrete("",labels=c("Three month before first wave","In first wave"))+scale_y_continuous("AUROC")
ggsave("/ModelleCovid/6monate/AUROC Veränderung aufgeteilt nach Modellen.jpg")
graph2ppt(file="/ModelleCovid/6monate/AUROC Veränderung aufgeteilt nach Modellen.pptx",margins = c(top = 0, right = 0, bottom = 0, left = 0),scaling = 200,aspectr=1)

ggplot(data=Leaderboard_Pandemie_long,aes(x=Zeitpunkt,y=PR,color=modelname)) +geom_line(aes(group=modelname))+geom_point(size=4,shape=21,stroke=2,fill="white") + theme_bw() +facet_grid(~algo) + theme(legend.position = "none",panel.grid = element_blank(),axis.text = element_text(size=16,color="black"),axis.title=element_text(size=18),strip.background = element_rect(fill="white"),strip.text = element_text(size=18))+scale_x_discrete("",labels=c("Three month before first wave","In first wave"))+scale_y_continuous("AUPR")
ggsave("/ModelleCovid/6monate/AUPR Veränderung aufgeteilt nach Modellen.jpg")
graph2ppt(file="/ModelleCovid/6monate/AUPR Veränderung aufgeteilt nach Modellen.pptx",margins = c(top = 0, right = 0, bottom = 0, left = 0),scaling = 200,aspectr=1)




################################################################################################################








train_vorPandemie           <- miceadds::load.data(file="Mortality Covid AutoMtrain_vorPandemie.RData",type="RData")
test_vorPandemie            <- miceadds::load.data(file="Mortality Covid AutoMtest_vorPandemie.RData",type="RData")
vorPandemie_validierung     <- miceadds::load.data(file="Mortality Covid AutoMvalid_vorPandemie.RData",type="RData")
pandemie                    <- miceadds::load.data(file="Mortality Covid AutoMpandemie.RData",type="RData")



train_vorPandemie$set <- "train"       
test_vorPandemie$set <- "test"       
vorPandemie_validierung$set <- "vorPandemie_validierung"    
pandemie$set <- "pandemie"                   


whole_df <- rbind(train_vorPandemie,
                  test_vorPandemie,
                  vorPandemie_validierung,
                  pandemie)

library(dataPreparation)
whole_df       <- set_col_as_factor(whole_df, cols = "auto", n_levels = 53, verbose = TRUE)
whole_df       <- data.frame(whole_df)

whole_df_num  <- whole_df        %>% select_if(is.numeric)


whole_df_fac      <- whole_df      %>% select(!all_of(names(whole_df_num      )))


whole_df_num      <- scale(whole_df_num      )


whole_df_num      <- as.data.frame(whole_df_num      )




whole_df      <- cbind(whole_df_num      ,whole_df_fac      )

train_vorPandemie <- whole_df %>% filter(set=="train") %>% select(-set)
test_vorPandemie <- whole_df %>% filter(set=="test") %>% select(-set)
vorPandemie_validierung <- whole_df %>% filter(set=="vorPandemie_validierung") %>% select(-set)
pandemie <- whole_df %>% filter(set=="pandemie") %>% select(-set)



train_vorPandemie_h20 <- as.h2o(train_vorPandemie)
test_vorPandemie_h20  <- as.h2o(test_vorPandemie)
valid_vorPandemie_h20 <- as.h2o(vorPandemie_validierung)
pandemie_h20          <- as.h2o(pandemie)


# Identify predictors and response
y <- "verstorben"

x <- setdiff(names(train_vorPandemie_h20), y)
#x <- setdiff(x,"gewichtungs_modell")

# For binary classification, response should be a factor
train_vorPandemie_h20[, y] <-  as.factor(train_vorPandemie_h20[, y])
test_vorPandemie_h20[, y]  <- as.factor(test_vorPandemie_h20[, y])
valid_vorPandemie_h20[, y] <- as.factor(valid_vorPandemie_h20[, y])
pandemie_h20[, y]          <- as.factor(pandemie_h20[, y])


# Run AutoML testrun 
aml_pandemie_14 <- h2o.automl(x = x, y = y,
                              training_frame = train_vorPandemie_h20,
                              #max_models = 2,
                              seed = 1,
                              validation_frame=test_vorPandemie_h20,
                              blending_frame=test_vorPandemie_h20,
                              balance_classes=TRUE,
                              stopping_metric="AUCPR",
                              sort_metric="AUCPR",
                              stopping_rounds=100,
                              leaderboard_frame = test_vorPandemie_h20,
                              nfolds=0,
                              max_runtime_secs=20*60*60,
                              #exclude_algos="GLM",
                              include_algos = c("GLM", "DeepLearning", "DRF","XGBoost","StackedEnsemble","GBM","DRF"),
                              preprocessing= 'target_encoding' )


save(aml_pandemie_14,file="/ModelleCovid/Skaliert/aml_pandemie_14.RData")

#aml_pandemie <- miceadds::load.data(file="aml_pandemie_val.RData",type="RData")
#aml_test_pandemie <- miceadds::load.data(file="aml_test_pandemie.RData",type="RData")
#ll <- miceadds::load.data(file="aml_test_pandemie.RData",type="RData")


####
#  Leaderboard Test
####

lb_vorPandemie_h20 <- h2o.get_leaderboard(object = aml_pandemie_14, extra_columns = "ALL")
lb_vorPandemie_h20
save(lb_vorPandemie_h20,file="/ModelleCovid/Skaliert/lb_vorPandemie_h20.RData")

###
#  Get Models
###

model <- as.vector(aml_pandemie_14@leaderboard$model_id)
model

###
#  Leaderboard in Panemie
###

# Alle Modelle vorhersagen











lb_vorPandemie_h20_df <- as.data.frame(lb_vorPandemie_h20)

lb_vorPandemie_h20_df_Stacked      <- lb_vorPandemie_h20_df %>% filter(algo=="StackedEnsemble")
lb_vorPandemie_h20_df_DeepLearning <- lb_vorPandemie_h20_df %>% filter(algo=="DeepLearning")
lb_vorPandemie_h20_df_DRF          <- lb_vorPandemie_h20_df %>% filter(algo=="DRF")
lb_vorPandemie_h20_df_GBM          <- lb_vorPandemie_h20_df %>% filter(algo=="GBM")
lb_vorPandemie_h20_df_GLM          <- lb_vorPandemie_h20_df %>% filter(algo=="GLM")
lb_vorPandemie_h20_df_XGBoost      <- lb_vorPandemie_h20_df %>% filter(algo=="XGBoost")

openxlsx::write.xlsx(rbind(lb_vorPandemie_h20_df[1,],
                           lb_vorPandemie_h20_df_Stacked[1,],     
                           lb_vorPandemie_h20_df_DeepLearning[1,],
                           lb_vorPandemie_h20_df_DRF[1,],         
                           lb_vorPandemie_h20_df_GBM[1,],         
                           lb_vorPandemie_h20_df_GLM[1,],         
                           lb_vorPandemie_h20_df_XGBoost[1,]     ),"/ModelleCovid/Skaliert/besteAlgos14.xlsx")





###########################################################################################################################################################################
#################################                                   Leaderboard in pandemie                               #################################################
###########################################################################################################################################################################
pr_ci <- function(x,y){
  nu <- log(x$auc.integral/(1-x$auc.integral))
  tau <- 1/sqrt(y*x$auc.integral*(1-x$auc.integral))
  lower <- exp(nu-1.96*tau)/(1+exp(nu-1.96*tau))
  lower
  upper <- exp(nu+1.96*tau)/(1+exp(nu+1.96*tau))
  upper
  return(paste(round(x$auc.integral,3)," [",round(lower,3)," to ",round(upper,3),"]",sep=""))
  
}
###############################################################################################################################################################################




library(pROC)
library(PRROC)

predictions_vor_pandemie <- data.frame(truth=vorPandemie_validierung$verstorben)
predictions_in_pandemie <- data.frame(truth=pandemie$verstorben)
Leaderboard_Pandemie <- data.frame(modelname=NA,
                                   algo=NA,
                                   AUC_vor_Pandemie=NA,
                                   AUC_mitCI_vor_Pandemie=NA,
                                   PR_vor_Pandemie=NA,
                                   PR_mitCI_vor_Pandemie=NA,
                                   AUC_in_Pandemie=NA,
                                   AUC_mitCI_in_Pandemie=NA,
                                   PR_in_Pandemie=NA,
                                   PR_mitCI_in_Pandemie=NA)

for(i in 1:length(model))
{
  model_loop <-h2o.getModel(model[i])
  
  h2o.saveModel(object = model_loop, path = "/ModelleCovid/Skaliert/Modellsicherung/", force = TRUE,filename=paste("Mortality_covid_",model[i],sep=""))
  # Predcit vor Pandemie
  
  p_t_vor_pandemie  <- h2o.predict(model_loop, newdata = valid_vorPandemie_h20)
  p_t_in_pandemie   <- h2o.predict(model_loop, newdata = pandemie_h20)
  
  # in Dataframe umwandeln
  
  p_t_vor_pandemie_df  <- as.data.frame(p_t_vor_pandemie  )
  p_t_in_pandemie_df   <- as.data.frame(p_t_in_pandemie   )
  
  
  
  
  predictions_vor_pandemie[,i+1] <- p_t_vor_pandemie_df$p1
  predictions_in_pandemie[,i+1]  <- p_t_in_pandemie_df$p1
  
  # AUC
  aaa_vor_pandemie <- pROC::ci.auc(predictions_vor_pandemie$truth,p_t_vor_pandemie_df$p1)
  aaa_in_pandemie  <- pROC::ci.auc(predictions_in_pandemie$truth,p_t_in_pandemie_df$p1)
  
  # PR
  
  pr_vor_pandemie <- pr.curve(scores.class0 = predictions_vor_pandemie[,i+1][predictions_vor_pandemie$truth==1],  predictions_vor_pandemie[,i+1][predictions_vor_pandemie$truth==0], curve = TRUE)
  pr_in_pandemie  <- pr.curve(scores.class0 = predictions_in_pandemie[,i+1][predictions_in_pandemie$truth==1],  predictions_in_pandemie[,i+1][predictions_in_pandemie$truth==0], curve = TRUE)
  
  
  pr_ci_vor_pandemie <- pr_ci(pr_vor_pandemie,length(predictions_vor_pandemie$truth))
  pr_ci_in_pandemie  <- pr_ci(pr_in_pandemie,length(predictions_in_pandemie$truth))
  
  Leaderboard_Pandemie <- rbind(Leaderboard_Pandemie,
                                c(lb_vorPandemie_h20[i,1][1],lb_vorPandemie_h20[i,10][1],aaa_vor_pandemie[2],paste(sprintf("%.3f",aaa_vor_pandemie[2])," [",sprintf("%.3f",aaa_vor_pandemie[1])," to ",sprintf("%.3f",aaa_vor_pandemie[3]),"]",sep=""),
                                  pr_vor_pandemie$auc.integral,pr_ci_vor_pandemie,
                                  aaa_in_pandemie[2],paste(sprintf("%.3f",aaa_in_pandemie[2])," [",sprintf("%.3f",aaa_in_pandemie[1])," to ",sprintf("%.3f",aaa_in_pandemie[3]),"]",sep=""),
                                  pr_in_pandemie$auc.integral,pr_ci_in_pandemie))
  
  
  
  
  print(paste(i," of ",length(model),sep=""))  
  
}


openxlsx::write.xlsx(Leaderboard_Pandemie,file="/ModelleCovid/Skaliert/Leaderboard_Pandemie.xlsx")

Leaderboard_Pandemie <- Leaderboard_Pandemie %>% filter(is.na(modelname)==FALSE)

#Leaderboard_Pandemie <- openxlsx::read.xlsx("/ModelleCovid/Skaliert/Leaderboard_Pandemie.xlsx")
#Leaderboard_Pandemie# Leaderboard_Pandemie
###########################################################################################################################################################################
#################################                             Raussuchen der jeweils besten Algos                         #################################################
###########################################################################################################################################################################
Leaderboard_Pandemie$AUC_vor_Pandemie <- as.numeric(Leaderboard_Pandemie$AUC_vor_Pandemie)
Leaderboard_Pandemie$PR_vor_Pandemie  <- as.numeric(Leaderboard_Pandemie$PR_vor_Pandemie)
Leaderboard_Pandemie$AUC_in_Pandemie  <- as.numeric(Leaderboard_Pandemie$AUC_in_Pandemie)
Leaderboard_Pandemie$PR_in_Pandemie   <- as.numeric(Leaderboard_Pandemie$PR_in_Pandemie)

Leaderboard_Pandemie <- Leaderboard_Pandemie %>% arrange(desc(AUC_vor_Pandemie)) %>% mutate(Reihenfolge.AUC.vor=1:n())
Leaderboard_Pandemie <- Leaderboard_Pandemie %>% arrange(desc(PR_vor_Pandemie)) %>% mutate(Reihenfolge.PR.vor=1:n())
Leaderboard_Pandemie <- Leaderboard_Pandemie %>% arrange(desc(AUC_in_Pandemie)) %>% mutate(Reihenfolge.AUC.in=1:n())
Leaderboard_Pandemie <- Leaderboard_Pandemie %>% arrange(desc(PR_in_Pandemie)) %>% mutate(Reihenfolge.PR.in=1:n())



lb_Stacked      <- Leaderboard_Pandemie %>% filter(algo=="StackedEnsemble")
lb_DeepLearning <- Leaderboard_Pandemie %>% filter(algo=="DeepLearning")
lb_DRF          <- Leaderboard_Pandemie %>% filter(algo=="DRF")
lb_GBM          <- Leaderboard_Pandemie %>% filter(algo=="GBM")
lb_GLM          <- Leaderboard_Pandemie %>% filter(algo=="GLM")
lb_XGBoost      <- Leaderboard_Pandemie %>% filter(algo=="XGBoost")




lb_Stacked      <- lb_Stacked     %>% arrange(desc(PR_vor_Pandemie))
lb_DeepLearning <- lb_DeepLearning%>% arrange(desc(PR_vor_Pandemie))
lb_DRF          <- lb_DRF         %>% arrange(desc(PR_vor_Pandemie))
lb_GBM          <- lb_GBM         %>% arrange(desc(PR_vor_Pandemie))
lb_GLM          <- lb_GLM         %>% arrange(desc(PR_vor_Pandemie))
lb_XGBoost      <- lb_XGBoost     %>% arrange(desc(PR_vor_Pandemie))

beste_vor_pandemie <- rbind(lb_Stacked     [1,],
                            lb_DeepLearning[1,],
                            lb_DRF         [1,],
                            lb_GBM         [1,],
                            lb_GLM         [1,],
                            lb_XGBoost     [1,])

beste_vor_pandemie %>% arrange(desc(PR_vor_Pandemie))


lb_Stacked      <- lb_Stacked     %>% arrange(desc(PR_in_Pandemie))
lb_DeepLearning <- lb_DeepLearning%>% arrange(desc(PR_in_Pandemie))
lb_DRF          <- lb_DRF         %>% arrange(desc(PR_in_Pandemie))
lb_GBM          <- lb_GBM         %>% arrange(desc(PR_in_Pandemie))
lb_GLM          <- lb_GLM         %>% arrange(desc(PR_in_Pandemie))
lb_XGBoost      <- lb_XGBoost     %>% arrange(desc(PR_in_Pandemie))

beste_in_pandemie <- rbind( lb_Stacked     [1,],
                            lb_DeepLearning[1,],
                            lb_DRF         [1,],
                            lb_GBM         [1,],
                            lb_GLM         [1,],
                            lb_XGBoost     [1,])

beste_in_pandemie %>% arrange(desc(PR_in_Pandemie))

openxlsx::write.xlsx(beste_vor_pandemie,"/ModelleCovid/Skaliert/besteAlgos_vor_pandemie.xlsx")
openxlsx::write.xlsx(beste_in_pandemie,"/ModelleCovid/Skaliert/besteAlgos_in_pandemie.xlsx")


###########################################################################################################################################################################
#################################                                           Kurven                                        #################################################
###########################################################################################################################################################################
####
#   PR mit besten vor Pandemie
####


best_vor_Pandemie_Stacked      <- lb_Stacked     %>% arrange(desc(PR_vor_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_vor_Pandemie_DeepLearning <- lb_DeepLearning%>% arrange(desc(PR_vor_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_vor_Pandemie_DRF          <- lb_DRF         %>% arrange(desc(PR_vor_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_vor_Pandemie_GBM          <- lb_GBM         %>% arrange(desc(PR_vor_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_vor_Pandemie_GLM          <- lb_GLM         %>% arrange(desc(PR_vor_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_vor_Pandemie_XGBoost      <- lb_XGBoost     %>% arrange(desc(PR_vor_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)


#StackedEnsemble_BestOfFamily_6_AutoML_1_20221205_81006 <- h2o.loadModel("/ModelleCovid/Skaliert/Modellsicherung/Mortality_covid_StackedEnsemble_BestOfFamily_6_AutoML_1_20221205_81006")


pr_vor_Panemie_Stacked      <- data.frame(pr.curve(scores.class0 = predictions_vor_pandemie[,which(model==best_vor_Pandemie_Stacked     $modelname)+1][predictions_vor_pandemie$truth==1],  predictions_vor_pandemie[,which(model==best_vor_Pandemie_Stacked     $modelname)+1][predictions_vor_pandemie$truth==0], curve = TRUE)$curve)
pr_vor_Panemie_DeepLearning <- data.frame(pr.curve(scores.class0 = predictions_vor_pandemie[,which(model==best_vor_Pandemie_DeepLearning$modelname)+1][predictions_vor_pandemie$truth==1],  predictions_vor_pandemie[,which(model==best_vor_Pandemie_DeepLearning$modelname)+1][predictions_vor_pandemie$truth==0], curve = TRUE)$curve)
pr_vor_Panemie_DRF          <- data.frame(pr.curve(scores.class0 = predictions_vor_pandemie[,which(model==best_vor_Pandemie_DRF         $modelname)+1][predictions_vor_pandemie$truth==1],  predictions_vor_pandemie[,which(model==best_vor_Pandemie_DRF         $modelname)+1][predictions_vor_pandemie$truth==0], curve = TRUE)$curve)
pr_vor_Panemie_GBM          <- data.frame(pr.curve(scores.class0 = predictions_vor_pandemie[,which(model==best_vor_Pandemie_GBM         $modelname)+1][predictions_vor_pandemie$truth==1],  predictions_vor_pandemie[,which(model==best_vor_Pandemie_GBM         $modelname)+1][predictions_vor_pandemie$truth==0], curve = TRUE)$curve)
pr_vor_Panemie_GLM          <- data.frame(pr.curve(scores.class0 = predictions_vor_pandemie[,which(model==best_vor_Pandemie_GLM         $modelname)+1][predictions_vor_pandemie$truth==1],  predictions_vor_pandemie[,which(model==best_vor_Pandemie_GLM         $modelname)+1][predictions_vor_pandemie$truth==0], curve = TRUE)$curve)
pr_vor_Panemie_XGBoost      <- data.frame(pr.curve(scores.class0 = predictions_vor_pandemie[,which(model==best_vor_Pandemie_XGBoost     $modelname)+1][predictions_vor_pandemie$truth==1],  predictions_vor_pandemie[,which(model==best_vor_Pandemie_XGBoost     $modelname)+1][predictions_vor_pandemie$truth==0], curve = TRUE)$curve)



pr_vor_Panemie_Stacked     $dataset <- "Stacked Ensemble"
pr_vor_Panemie_DeepLearning$dataset <- "DeepLearning"
pr_vor_Panemie_DRF         $dataset <- "Deep random forest"
pr_vor_Panemie_GBM         $dataset <- "Gradient boosting machine"
pr_vor_Panemie_GLM         $dataset <- "Generalized linear model"
pr_vor_Panemie_XGBoost     $dataset <- "XGboost"





pr_curve_df <- rbind(pr_vor_Panemie_Stacked     ,
                     pr_vor_Panemie_DeepLearning,
                     pr_vor_Panemie_DRF         ,
                     pr_vor_Panemie_GBM         ,
                     pr_vor_Panemie_GLM         ,
                     pr_vor_Panemie_XGBoost     )

names(pr_curve_df)
names(pr_curve_df) <- c("Recall","Precision","threshold","dataset")

pr_curve_df <- pr_curve_df %>% arrange( dataset,desc(Precision),Recall)


ggplot(data=pr_curve_df ,aes(x=Recall,y=Precision,color=dataset))+
  geom_line(linewidth=2)+
  theme_bw()+
  geom_hline(yintercept = 0.01020408 ,linetype=2)+
  scale_y_continuous(lim=c(0,1),expand = c(0.01, 0.01))+
  scale_x_continuous(expand = c(0.01, 0.01)) +
  theme(axis.title = element_text(size=20),
        axis.text = element_text(size=18,color="black"),
        legend.text = element_text(size=18),
        legend.title = element_text(size=20),
        panel.grid = element_blank(),
        legend.position = c(.8,.8))+
  scale_color_manual("",values=c("#00BF7D","#00B4C5","#0073E6","#B3C7F7","#F57600","#B51963"))
graph2ppt(file="/ModelleCovid/Skaliert/PowerPointGrafiken/PR beste Algorihmen vor Pandemie.pptx")


###
#  AUC
###

auc_vor_Panemie_Stacked      <- data.frame(sensitivities=pROC::roc(predictions_vor_pandemie$truth,predictions_vor_pandemie[,which(model==best_vor_Pandemie_Stacked     $modelname)+1])$sensitivities,eins_minus_specificities=1-pROC::roc(predictions_vor_pandemie$truth,predictions_vor_pandemie[,which(model==best_vor_Pandemie_Stacked     $modelname)+1])$specificities)
auc_vor_Panemie_DeepLearning <- data.frame(sensitivities=pROC::roc(predictions_vor_pandemie$truth,predictions_vor_pandemie[,which(model==best_vor_Pandemie_DeepLearning$modelname)+1])$sensitivities,eins_minus_specificities=1-pROC::roc(predictions_vor_pandemie$truth,predictions_vor_pandemie[,which(model==best_vor_Pandemie_DeepLearning$modelname)+1])$specificities)
auc_vor_Panemie_DRF          <- data.frame(sensitivities=pROC::roc(predictions_vor_pandemie$truth,predictions_vor_pandemie[,which(model==best_vor_Pandemie_DRF         $modelname)+1])$sensitivities,eins_minus_specificities=1-pROC::roc(predictions_vor_pandemie$truth,predictions_vor_pandemie[,which(model==best_vor_Pandemie_DRF         $modelname)+1])$specificities)
auc_vor_Panemie_GBM          <- data.frame(sensitivities=pROC::roc(predictions_vor_pandemie$truth,predictions_vor_pandemie[,which(model==best_vor_Pandemie_GBM         $modelname)+1])$sensitivities,eins_minus_specificities=1-pROC::roc(predictions_vor_pandemie$truth,predictions_vor_pandemie[,which(model==best_vor_Pandemie_GBM         $modelname)+1])$specificities)
auc_vor_Panemie_GLM          <- data.frame(sensitivities=pROC::roc(predictions_vor_pandemie$truth,predictions_vor_pandemie[,which(model==best_vor_Pandemie_GLM         $modelname)+1])$sensitivities,eins_minus_specificities=1-pROC::roc(predictions_vor_pandemie$truth,predictions_vor_pandemie[,which(model==best_vor_Pandemie_GLM         $modelname)+1])$specificities)
auc_vor_Panemie_XGBoost      <- data.frame(sensitivities=pROC::roc(predictions_vor_pandemie$truth,predictions_vor_pandemie[,which(model==best_vor_Pandemie_XGBoost     $modelname)+1])$sensitivities,eins_minus_specificities=1-pROC::roc(predictions_vor_pandemie$truth,predictions_vor_pandemie[,which(model==best_vor_Pandemie_XGBoost     $modelname)+1])$specificities)


auc_vor_Panemie_Stacked     $dataset <- "Stacked Ensemble"
auc_vor_Panemie_DeepLearning$dataset <- "DeepLearning"
auc_vor_Panemie_DRF         $dataset <- "Deep random forest"
auc_vor_Panemie_GBM         $dataset <- "Gradient boosting machine"
auc_vor_Panemie_GLM         $dataset <- "Generalized linear model"
auc_vor_Panemie_XGBoost     $dataset <- "XGboost"





auc_curve_vor_Pandemie_df <- rbind(auc_vor_Panemie_Stacked     ,
                                   auc_vor_Panemie_DeepLearning,
                                   auc_vor_Panemie_DRF         ,
                                   auc_vor_Panemie_GBM         ,
                                   auc_vor_Panemie_GLM         ,
                                   auc_vor_Panemie_XGBoost     )

names(auc_curve_vor_Pandemie_df)


pr_curve_df <- pr_curve_df %>% arrange( dataset,desc(Precision),Recall)


ggplot(data=auc_curve_vor_Pandemie_df ,aes(x=eins_minus_specificities,y=sensitivities,color=dataset))+
  geom_line(linewidth=2)+
  theme_bw()+
  #geom_hline(yintercept = 0.01020408 ,linetype=2)+
  scale_y_continuous(lim=c(0,1),expand = c(0.01, 0.01),"Sensitivity")+
  scale_x_continuous(expand = c(0.01, 0.01),"1-Specificity") +
  theme(axis.title = element_text(size=20),
        axis.text = element_text(size=18,color="black"),
        legend.text = element_text(size=18),
        legend.title = element_text(size=20),
        panel.grid = element_blank(),
        legend.position = c(.8,.5))+
  scale_color_manual("",values=c("#00BF7D","#00B4C5","#0073E6","#B3C7F7","#F57600","#B51963"))
graph2ppt(file="/ModelleCovid/Skaliert/PowerPointGrafiken/AUC beste Algorihmen vor Pandemie.pptx")



####
#   PR mit besten in Pandemie
####



best_in_Pandemie_Stacked      <- lb_Stacked     %>% arrange(desc(PR_in_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_in_Pandemie_DeepLearning <- lb_DeepLearning%>% arrange(desc(PR_in_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_in_Pandemie_DRF          <- lb_DRF         %>% arrange(desc(PR_in_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_in_Pandemie_GBM          <- lb_GBM         %>% arrange(desc(PR_in_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_in_Pandemie_GLM          <- lb_GLM         %>% arrange(desc(PR_in_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_in_Pandemie_XGBoost      <- lb_XGBoost     %>% arrange(desc(PR_in_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)





pr_in_Panemie_Stacked      <- data.frame(pr.curve(scores.class0 = predictions_in_pandemie[,which(model==best_in_Pandemie_Stacked     $modelname)+1][predictions_in_pandemie$truth==1],  predictions_in_pandemie[,which(model==best_in_Pandemie_Stacked     $modelname)+1][predictions_in_pandemie$truth==0], curve = TRUE)$curve)
pr_in_Panemie_DeepLearning <- data.frame(pr.curve(scores.class0 = predictions_in_pandemie[,which(model==best_in_Pandemie_DeepLearning$modelname)+1][predictions_in_pandemie$truth==1],  predictions_in_pandemie[,which(model==best_in_Pandemie_DeepLearning$modelname)+1][predictions_in_pandemie$truth==0], curve = TRUE)$curve)
pr_in_Panemie_DRF          <- data.frame(pr.curve(scores.class0 = predictions_in_pandemie[,which(model==best_in_Pandemie_DRF         $modelname)+1][predictions_in_pandemie$truth==1],  predictions_in_pandemie[,which(model==best_in_Pandemie_DRF         $modelname)+1][predictions_in_pandemie$truth==0], curve = TRUE)$curve)
pr_in_Panemie_GBM          <- data.frame(pr.curve(scores.class0 = predictions_in_pandemie[,which(model==best_in_Pandemie_GBM         $modelname)+1][predictions_in_pandemie$truth==1],  predictions_in_pandemie[,which(model==best_in_Pandemie_GBM         $modelname)+1][predictions_in_pandemie$truth==0], curve = TRUE)$curve)
pr_in_Panemie_GLM          <- data.frame(pr.curve(scores.class0 = predictions_in_pandemie[,which(model==best_in_Pandemie_GLM         $modelname)+1][predictions_in_pandemie$truth==1],  predictions_in_pandemie[,which(model==best_in_Pandemie_GLM         $modelname)+1][predictions_in_pandemie$truth==0], curve = TRUE)$curve)
pr_in_Panemie_XGBoost      <- data.frame(pr.curve(scores.class0 = predictions_in_pandemie[,which(model==best_in_Pandemie_XGBoost     $modelname)+1][predictions_in_pandemie$truth==1],  predictions_in_pandemie[,which(model==best_in_Pandemie_XGBoost     $modelname)+1][predictions_in_pandemie$truth==0], curve = TRUE)$curve)



pr_in_Panemie_Stacked     $dataset <- "Stacked Ensemble"
pr_in_Panemie_DeepLearning$dataset <- "DeepLearning"
pr_in_Panemie_DRF         $dataset <- "Deep random forest"
pr_in_Panemie_GBM         $dataset <- "Gradient boosting machine"
pr_in_Panemie_GLM         $dataset <- "Generalized linear model"
pr_in_Panemie_XGBoost     $dataset <- "XGboost"





pr_curve_df <- rbind(pr_in_Panemie_Stacked     ,
                     pr_in_Panemie_DeepLearning,
                     pr_in_Panemie_DRF         ,
                     pr_in_Panemie_GBM         ,
                     pr_in_Panemie_GLM         ,
                     pr_in_Panemie_XGBoost     )

names(pr_curve_df)
names(pr_curve_df) <- c("Recall","Precision","threshold","dataset")

pr_curve_df <- pr_curve_df %>% arrange( dataset,desc(Precision),Recall)


ggplot(data=pr_curve_df ,aes(x=Recall,y=Precision,color=dataset))+
  geom_line(linewidth=2)+
  theme_bw()+
  geom_hline(yintercept = 0.01020408 ,linetype=2)+
  scale_y_continuous(lim=c(0,1),expand = c(0.01, 0.01))+
  scale_x_continuous(expand = c(0.01, 0.01)) +
  theme(axis.title = element_text(size=20),
        axis.text = element_text(size=18,color="black"),
        legend.text = element_text(size=18),
        legend.title = element_text(size=20),
        panel.grid = element_blank(),
        legend.position = c(.8,.8))+
  scale_color_manual("",values=c("#00BF7D","#00B4C5","#0073E6","#B3C7F7","#F57600","#B51963"))
graph2ppt(file="/ModelleCovid/Skaliert/PowerPointGrafiken/PR beste Algorihmen in Pandemie.pptx")


###
#  AUC
###

auc_in_Panemie_Stacked      <- data.frame(sensitivities=pROC::roc(predictions_in_pandemie$truth,predictions_in_pandemie[,which(model==best_in_Pandemie_Stacked     $modelname)+1])$sensitivities,eins_minus_specificities=1-pROC::roc(predictions_in_pandemie$truth,predictions_in_pandemie[,which(model==best_in_Pandemie_Stacked     $modelname)+1])$specificities)
auc_in_Panemie_DeepLearning <- data.frame(sensitivities=pROC::roc(predictions_in_pandemie$truth,predictions_in_pandemie[,which(model==best_in_Pandemie_DeepLearning$modelname)+1])$sensitivities,eins_minus_specificities=1-pROC::roc(predictions_in_pandemie$truth,predictions_in_pandemie[,which(model==best_in_Pandemie_DeepLearning$modelname)+1])$specificities)
auc_in_Panemie_DRF          <- data.frame(sensitivities=pROC::roc(predictions_in_pandemie$truth,predictions_in_pandemie[,which(model==best_in_Pandemie_DRF         $modelname)+1])$sensitivities,eins_minus_specificities=1-pROC::roc(predictions_in_pandemie$truth,predictions_in_pandemie[,which(model==best_in_Pandemie_DRF         $modelname)+1])$specificities)
auc_in_Panemie_GBM          <- data.frame(sensitivities=pROC::roc(predictions_in_pandemie$truth,predictions_in_pandemie[,which(model==best_in_Pandemie_GBM         $modelname)+1])$sensitivities,eins_minus_specificities=1-pROC::roc(predictions_in_pandemie$truth,predictions_in_pandemie[,which(model==best_in_Pandemie_GBM         $modelname)+1])$specificities)
auc_in_Panemie_GLM          <- data.frame(sensitivities=pROC::roc(predictions_in_pandemie$truth,predictions_in_pandemie[,which(model==best_in_Pandemie_GLM         $modelname)+1])$sensitivities,eins_minus_specificities=1-pROC::roc(predictions_in_pandemie$truth,predictions_in_pandemie[,which(model==best_in_Pandemie_GLM         $modelname)+1])$specificities)
auc_in_Panemie_XGBoost      <- data.frame(sensitivities=pROC::roc(predictions_in_pandemie$truth,predictions_in_pandemie[,which(model==best_in_Pandemie_XGBoost     $modelname)+1])$sensitivities,eins_minus_specificities=1-pROC::roc(predictions_in_pandemie$truth,predictions_in_pandemie[,which(model==best_in_Pandemie_XGBoost     $modelname)+1])$specificities)


auc_in_Panemie_Stacked     $dataset <- "Stacked Ensemble"
auc_in_Panemie_DeepLearning$dataset <- "DeepLearning"
auc_in_Panemie_DRF         $dataset <- "Deep random forest"
auc_in_Panemie_GBM         $dataset <- "Gradient boosting machine"
auc_in_Panemie_GLM         $dataset <- "Generalized linear model"
auc_in_Panemie_XGBoost     $dataset <- "XGboost"





auc_curve_in_Pandemie_df <- rbind(auc_in_Panemie_Stacked     ,
                                  auc_in_Panemie_DeepLearning,
                                  auc_in_Panemie_DRF         ,
                                  auc_in_Panemie_GBM         ,
                                  auc_in_Panemie_GLM         ,
                                  auc_in_Panemie_XGBoost     )

names(auc_curve_in_Pandemie_df)


pr_curve_df <- pr_curve_df %>% arrange( dataset,desc(Precision),Recall)


ggplot(data=auc_curve_in_Pandemie_df ,aes(x=eins_minus_specificities,y=sensitivities,color=dataset))+
  geom_line(linewidth=2)+
  theme_bw()+
  #geom_hline(yintercept = 0.01020408 ,linetype=2)+
  scale_y_continuous(lim=c(0,1),expand = c(0.01, 0.01),"Sensitivity")+
  scale_x_continuous(expand = c(0.01, 0.01),"1-Specificity") +
  theme(axis.title = element_text(size=20),
        axis.text = element_text(size=18,color="black"),
        legend.text = element_text(size=18),
        legend.title = element_text(size=20),
        panel.grid = element_blank(),
        legend.position = c(.8,.5))+
  scale_color_manual("",values=c("#00BF7D","#00B4C5","#0073E6","#B3C7F7","#F57600","#B51963"))
graph2ppt(file="/ModelleCovid/Skaliert/PowerPointGrafiken/AUC beste Algorihmen in Pandemie.pptx")



###########################################################################################################################################################################
#################################                                         Importance                                      #################################################
###########################################################################################################################################################################

best_vor_Pandemie_Stacked      <- lb_Stacked     %>% arrange(desc(PR_vor_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_vor_Pandemie_DeepLearning <- lb_DeepLearning%>% arrange(desc(PR_vor_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_vor_Pandemie_DRF          <- lb_DRF         %>% arrange(desc(PR_vor_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_vor_Pandemie_GBM          <- lb_GBM         %>% arrange(desc(PR_vor_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_vor_Pandemie_GLM          <- lb_GLM         %>% arrange(desc(PR_vor_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_vor_Pandemie_XGBoost      <- lb_XGBoost     %>% arrange(desc(PR_vor_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)



best_stacked      <- h2o.loadModel(paste("/ModelleCovid/Skaliert/Modellsicherung/Mortality_covid_",best_vor_Pandemie_Stacked$modelname,sep=""))
best_DeepLearning <- h2o.loadModel(paste("/ModelleCovid/Skaliert/Modellsicherung/Mortality_covid_",best_vor_Pandemie_DeepLearning$modelname,sep=""))
best_DR           <- h2o.loadModel(paste("/ModelleCovid/Skaliert/Modellsicherung/Mortality_covid_",best_vor_Pandemie_DRF         $modelname,sep=""))
best_GBM          <- h2o.loadModel(paste("/ModelleCovid/Skaliert/Modellsicherung/Mortality_covid_",best_vor_Pandemie_GBM         $modelname,sep=""))
best_GLM          <- h2o.loadModel(paste("/ModelleCovid/Skaliert/Modellsicherung/Mortality_covid_",best_vor_Pandemie_GLM         $modelname,sep=""))
best_XGBoost      <- h2o.loadModel(paste("/ModelleCovid/Skaliert/Modellsicherung/Mortality_covid_",best_vor_Pandemie_XGBoost     $modelname,sep=""))



vi_vor_Pandemie_Stacked     <- h2o.permutation_importance(h2o.getModel(best_vor_Pandemie_Stacked$modelname), newdata = vorPandemie_validierung)
vi_vor_Pandemie_DeepLearning<- h2o.varimp(h2o.getModel(best_vor_Pandemie_DeepLearning$modelname))
vi_vor_Pandemie_DRF         <- h2o.varimp(h2o.getModel(best_vor_Pandemie_DRF         $modelname))
vi_vor_Pandemie_GBM         <- h2o.varimp(h2o.getModel(best_vor_Pandemie_GBM         $modelname))
vi_vor_Pandemie_GLM         <- h2o.varimp(h2o.getModel(best_vor_Pandemie_GLM         $modelname))
vi_vor_Pandemie_XGBoost     <- h2o.varimp(h2o.getModel(best_vor_Pandemie_XGBoost     $modelname))


write.csv2(vi_vor_Pandemie_Stacked     ,file="/ModelleCovid/Skaliert/vi_vor_Pandemie_Stacked.csv",sep=",",dec=";")
write.csv2(vi_vor_Pandemie_DeepLearning,file="/ModelleCovid/Skaliert/vi_vor_Pandemie_DeepLearning.csv",sep=",",dec=";")
write.csv2(vi_vor_Pandemie_DRF         ,file="/ModelleCovid/Skaliert/vi_vor_Pandemie_DRF.csv",sep=",",dec=";")
write.csv2(vi_vor_Pandemie_GBM         ,file="/ModelleCovid/Skaliert/vi_vor_Pandemie_GBM.csv",sep=",",dec=";")
write.csv2(vi_vor_Pandemie_GLM         ,file="/ModelleCovid/Skaliert/vi_vor_Pandemie_GLM.csv",sep=",",dec=";")
write.csv2(vi_vor_Pandemie_XGBoost     ,file="/ModelleCovid/Skaliert/vi_vor_Pandemie_XGBoost.csv",sep=",",dec=";")

vi_vor_Pandemie_Stacked     <- read.csv(file="/ModelleCovid/Skaliert/vi_vor_Pandemie_Stacked.csv",sep=";",dec=",")
vi_vor_Pandemie_DeepLearning<- read.csv(file="/ModelleCovid/Skaliert/vi_vor_Pandemie_DeepLearning.csv",sep=";",dec=",")
vi_vor_Pandemie_DRF         <- read.csv(file="/ModelleCovid/Skaliert/vi_vor_Pandemie_DRF.csv",sep=";",dec=",")
vi_vor_Pandemie_GBM         <- read.csv(file="/ModelleCovid/Skaliert/vi_vor_Pandemie_GBM.csv",sep=";",dec=",")
vi_vor_Pandemie_GLM         <- read.csv(file="/ModelleCovid/Skaliert/vi_vor_Pandemie_GLM.csv",sep=";",dec=",")
vi_vor_Pandemie_XGBoost     <- read.csv(file="/ModelleCovid/Skaliert/vi_vor_Pandemie_XGBoost.csv",sep=";",dec=",")


vi_vor_pandemie <- list(Stacked_Ensemble=v1_stacked $Variable[1:100]    ,
                        DeepLearning=vi_vor_Pandemie_DeepLearning$variable[1:100],
                        DRF=vi_vor_Pandemie_DRF$variable[1:100]        ,
                        GBM=vi_vor_Pandemie_GBM$variable[1:100]        ,
                        GLM=vi_vor_Pandemie_GLM$variable[1:100]         ,
                        XGboost=vi_vor_Pandemie_XGBoost$variable[1:100]     )


vd <- ggVennDiagram(vi_vor_pandemie)
ggVennDiagram(vi_vor_pandemie,show_intersect = F,label_color = "white",label="count")
graph2ppt(file="/ModelleCovid/Skaliert/PowerPointGrafiken/Venn_Pandemie.pptx")




## in Pandemie




best_in_Pandemie_Stacked      <- Leaderboard_Pandemie %>% filter(algo=="StackedEnsemble")  %>% arrange(desc(PR_in_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_in_Pandemie_DeepLearning <- Leaderboard_Pandemie %>% filter(algo=="DeepLearning")     %>% arrange(desc(PR_in_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_in_Pandemie_DRF          <- Leaderboard_Pandemie %>% filter(algo=="DRF")              %>% arrange(desc(PR_in_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_in_Pandemie_GBM          <- Leaderboard_Pandemie %>% filter(algo=="GBM")              %>% arrange(desc(PR_in_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_in_Pandemie_GLM          <- Leaderboard_Pandemie %>% filter(algo=="GLM")              %>% arrange(desc(PR_in_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_in_Pandemie_XGBoost      <- Leaderboard_Pandemie %>% filter(algo=="XGBoost")          %>% arrange(desc(PR_in_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)



best_stacked      <- h2o.loadModel(paste("/ModelleCovid/Skaliert/Modellsicherung/Mortality_covid_",best_in_Pandemie_Stacked$modelname,sep=""))
best_DeepLearning <- h2o.loadModel(paste("/ModelleCovid/Skaliert/Modellsicherung/Mortality_covid_",best_in_Pandemie_DeepLearning$modelname,sep=""))
best_DR           <- h2o.loadModel(paste("/ModelleCovid/Skaliert/Modellsicherung/Mortality_covid_",best_in_Pandemie_DRF         $modelname,sep=""))
best_GBM          <- h2o.loadModel(paste("/ModelleCovid/Skaliert/Modellsicherung/Mortality_covid_",best_in_Pandemie_GBM         $modelname,sep=""))
best_GLM          <- h2o.loadModel(paste("/ModelleCovid/Skaliert/Modellsicherung/Mortality_covid_",best_in_Pandemie_GLM         $modelname,sep=""))
best_XGBoost      <- h2o.loadModel(paste("/ModelleCovid/Skaliert/Modellsicherung/Mortality_covid_",best_in_Pandemie_XGBoost     $modelname,sep=""))


vi_in_Pandemie_Stacked          <- h2o.permutation_importance(h2o.getModel(best_vor_Pandemie_Stacked$modelname), newdata = pandemie_h20)
vi_in_Pandemie_DeepLearning     <- h2o.varimp(h2o.getModel(best_vor_Pandemie_DeepLearning$modelname))
vi_in_Pandemie_DRF              <- h2o.varimp(h2o.getModel(best_vor_Pandemie_DRF         $modelname))
vi_in_Pandemie_GBM              <- h2o.varimp(h2o.getModel(best_vor_Pandemie_GBM         $modelname))
vi_in_Pandemie_GLM              <- h2o.varimp(h2o.getModel(best_vor_Pandemie_GLM         $modelname))
vi_in_Pandemie_XGBoost          <- h2o.varimp(h2o.getModel(best_vor_Pandemie_XGBoost     $modelname))






write.csv2(vi_in_Pandemie_Stacked     ,file="/ModelleCovid/Skaliert/vi_in_Pandemie_Stacked.csv",sep=",",dec=";")
write.csv2(vi_in_Pandemie_DeepLearning,file="/ModelleCovid/Skaliert/vi_in_Pandemie_DeepLearning.csv",sep=",",dec=";")
write.csv2(vi_in_Pandemie_DRF         ,file="/ModelleCovid/Skaliert/vi_in_Pandemie_DRF.csv",sep=",",dec=";")
write.csv2(vi_in_Pandemie_GBM         ,file="/ModelleCovid/Skaliert/vi_in_Pandemie_GBM.csv",sep=",",dec=";")
write.csv2(vi_in_Pandemie_GLM         ,file="/ModelleCovid/Skaliert/vi_in_Pandemie_GLM.csv",sep=",",dec=";")
write.csv2(vi_in_Pandemie_XGBoost     ,file="/ModelleCovid/Skaliert/vi_in_Pandemie_XGBoost.csv",sep=",",dec=";")



####
# Extrahieren der Übereinstimmungen


Intersect <- function (x) {  
  # Multiple set version of intersect
  # x is a list
  if (length(x) == 1) {
    unlist(x)
  } else if (length(x) == 2) {
    intersect(x[[1]], x[[2]])
  } else if (length(x) > 2){
    intersect(x[[1]], Intersect(x[-1]))
  }
}

Union <- function (x) {  
  # Multiple set version of union
  # x is a list
  if (length(x) == 1) {
    unlist(x)
  } else if (length(x) == 2) {
    union(x[[1]], x[[2]])
  } else if (length(x) > 2) {
    union(x[[1]], Union(x[-1]))
  }
}

Setdiff <- function (x, y) {
  # Remove the union of the y's from the common x's. 
  # x and y are lists of characters.
  xx <- Intersect(x)
  yy <- Union(y)
  setdiff(xx, yy)
}

combs <- 
  unlist(lapply(1:length(vi_vor_pandemie), 
                function(j) combn(names(vi_vor_pandemie), j, simplify = FALSE)),
         recursive = FALSE)
names(combs) <- sapply(combs, function(i) paste0(i, collapse = ""))


elements <- 
  lapply(combs, function(i) Setdiff(vi_vor_pandemie[i], vi_vor_pandemie[setdiff(names(vi_vor_pandemie), i)]))

n.elements <- sapply(elements, length)
print(n.elements)

cbind.fill<-function(...){
  nm <- list(...) 
  nm<-lapply(nm, as.matrix)
  n <- max(sapply(nm, nrow)) 
  do.call(cbind, lapply(nm, function (x) 
    rbind(x, matrix(, n-nrow(x), ncol(x))))) 
}
df <-  as.data.frame(do.call(cbind.fill, elements))

names(df) <- names(elements)

openxlsx::write.xlsx(df,"/ModelleCovid/Skaliert/Übereinstimmungen Venn Diagramm.xlsx")


str(vd)

graph2ppt(file="/ModelleCovid/Skaliert/VennDiagramm Top100.pptx")

openxlsx::write.xlsx(Reduce(intersect, vi_vor_pandemie),"/ModelleCovid/Skaliert/Intersect Variablen aller Modelle.xlsx")


###########################################################################################################################################################################
#################################                                       Plot Veränderung                                  #################################################
###########################################################################################################################################################################

library(dplyr)
library(ggplot2)
library(ggridges)
library(export)


head(Leaderboard_Pandemie)


Leaderboard_Pandemie_vor <- Leaderboard_Pandemie %>% select(modelname,algo,AUC_vor_Pandemie,PR_vor_Pandemie) %>% rename(AUC=AUC_vor_Pandemie,PR=PR_vor_Pandemie)
Leaderboard_Pandemie_in  <- Leaderboard_Pandemie %>% select(modelname,algo,AUC_in_Pandemie,PR_in_Pandemie)   %>% rename(AUC=AUC_in_Pandemie,PR=PR_in_Pandemie)

Leaderboard_Pandemie_vor$Zeitpunkt <- "vor"
Leaderboard_Pandemie_in $Zeitpunkt <- "in"


Leaderboard_Pandemie_long <- rbind(Leaderboard_Pandemie_vor,
                                   Leaderboard_Pandemie_in )
Leaderboard_Pandemie_long <- na.omit(Leaderboard_Pandemie_long)

Leaderboard_Pandemie_long$Zeitpunkt <- factor(Leaderboard_Pandemie_long$Zeitpunkt,levels  =c("vor","in"))
Leaderboard_Pandemie_long$AUC <- as.numeric(as.character(Leaderboard_Pandemie_long$AUC))
Leaderboard_Pandemie_long$PR <- as.numeric(as.character(Leaderboard_Pandemie_long$PR))



ggplot(data=Leaderboard_Pandemie_long,aes(x=Zeitpunkt,y=AUC,color=modelname))+geom_line(aes(group=modelname))+geom_point(size=4,shape=21,stroke=2,fill="white") + theme_bw() + theme(legend.position = "none",panel.grid = element_blank(),axis.text = element_text(size=48,color="black"),axis.title=element_text(size=54))+scale_x_discrete("",labels=c("Three month before first wave","In first wave"))+scale_y_continuous("AUROC")
ggsave("/ModelleCovid/Skaliert/AUROC Veränderung.jpg")
graph2ppt(file="/ModelleCovid/Skaliert/AUROC Veränderung.pptx",margins = c(top = 0, right = 0, bottom = 0, left = 0),scaling = 200,aspectr=1)

ggplot(data=Leaderboard_Pandemie_long,aes(x=Zeitpunkt,y=PR,color=modelname)) +geom_line(aes(group=modelname))+geom_point(size=4,shape=21,stroke=2,fill="white") + theme_bw() + theme(legend.position = "none",panel.grid = element_blank(),axis.text = element_text(size=16,color="black"),axis.title=element_text(size=18))+scale_x_discrete("",labels=c("Three month before first wave","In first wave"))+scale_y_continuous("AUPR")
ggsave("/ModelleCovid/Skaliert/AUPR Veränderung.jpg")
graph2ppt(file="/ModelleCovid/Skaliert/AUPR Veränderung.pptx",margins = c(top = 0, right = 0, bottom = 0, left = 0),scaling = 200,aspectr=1)


ggplot(data=Leaderboard_Pandemie_long,aes(x=Zeitpunkt,y=AUC,color=modelname))+geom_line(aes(group=modelname))+geom_point(size=4,shape=21,stroke=2,fill="white") + theme_bw() +facet_grid(~algo) + theme(legend.position = "none",panel.grid = element_blank(),axis.text = element_text(size=16,color="black"),axis.title=element_text(size=18),strip.background = element_rect(fill="white"),strip.text = element_text(size=18))+scale_x_discrete("",labels=c("Three month before first wave","In first wave"))+scale_y_continuous("AUROC")
ggsave("/ModelleCovid/Skaliert/AUROC Veränderung aufgeteilt nach Modellen.jpg")
graph2ppt(file="/ModelleCovid/Skaliert/AUROC Veränderung aufgeteilt nach Modellen.pptx",margins = c(top = 0, right = 0, bottom = 0, left = 0),scaling = 200,aspectr=1)

ggplot(data=Leaderboard_Pandemie_long,aes(x=Zeitpunkt,y=PR,color=modelname)) +geom_line(aes(group=modelname))+geom_point(size=4,shape=21,stroke=2,fill="white") + theme_bw() +facet_grid(~algo) + theme(legend.position = "none",panel.grid = element_blank(),axis.text = element_text(size=16,color="black"),axis.title=element_text(size=18),strip.background = element_rect(fill="white"),strip.text = element_text(size=18))+scale_x_discrete("",labels=c("Three month before first wave","In first wave"))+scale_y_continuous("AUPR")
ggsave("/ModelleCovid/Skaliert/AUPR Veränderung aufgeteilt nach Modellen.jpg")
graph2ppt(file="/ModelleCovid/Skaliert/AUPR Veränderung aufgeteilt nach Modellen.pptx",margins = c(top = 0, right = 0, bottom = 0, left = 0),scaling = 200,aspectr=1)




################################################################################################################
### Weight



train_vorPandemie           <- miceadds::load.data(file="Mortality Covid AutoMtrain_vorPandemie.RData",type="RData")
test_vorPandemie            <- miceadds::load.data(file="Mortality Covid AutoMtest_vorPandemie.RData",type="RData")
vorPandemie_validierung     <- miceadds::load.data(file="Mortality Covid AutoMvalid_vorPandemie.RData",type="RData")
pandemie                    <- miceadds::load.data(file="Mortality Covid AutoMpandemie.RData",type="RData")

###
# Gewichte vergeben
###
train_vorPandemie$my <- paste(train_vorPandemie$month,train_vorPandemie$year,sep="_")
test_vorPandemie$my  <- paste(test_vorPandemie$month,test_vorPandemie$year,sep="_")

weight_df <- train_vorPandemie %>% select(month,year,my) %>% arrange(year,month) %>% distinct(my,.keep_all=T)

head(weight_df)
weight_df <- weight_df %>% mutate(weight=1:n()) %>% mutate(weight_max=max(weight)) %>% mutate(weight_proz=weight/weight_max)
head(weight_df)

summary(weight_df$weight_proz)

train_vorPandemie                          <- left_join(train_vorPandemie,weight_df %>% select(my,weight_proz) %>% rename(gewichtungs_modell=weight_proz),by="my")
test_vorPandemie                           <- left_join(test_vorPandemie ,weight_df %>% select(my,weight_proz) %>% rename(gewichtungs_modell=weight_proz),by="my")
vorPandemie_validierung$gewichtungs_modell <- 1
pandemie$gewichtungs_modell                <- 1               





train_vorPandemie_h20 <- as.h2o(train_vorPandemie)
test_vorPandemie_h20  <- as.h2o(test_vorPandemie)
valid_vorPandemie_h20 <- as.h2o(vorPandemie_validierung)
pandemie_h20          <- as.h2o(pandemie)


# Identify predictors and response
y <- "verstorben"
x <- setdiff(names(train_vorPandemie_h20), y)
x <- setdiff(x,"gewichtungs_modell")

# For binary classification, response should be a factor
train_vorPandemie_h20[, y] <-  as.factor(train_vorPandemie_h20[, y])
test_vorPandemie_h20[, y]  <- as.factor(test_vorPandemie_h20[, y])
valid_vorPandemie_h20[, y] <- as.factor(valid_vorPandemie_h20[, y])
pandemie_h20[, y]          <- as.factor(pandemie_h20[, y])


# Run AutoML testrun 
aml_pandemie_14 <- h2o.automl(x = x, y = y,
                              training_frame = train_vorPandemie_h20,
                              #max_models = 2,
                              seed = 1,
                              validation_frame=test_vorPandemie_h20,
                              blending_frame=test_vorPandemie_h20,
                              #balance_classes=TRUE,
                              stopping_metric="AUCPR",
                              sort_metric="AUCPR",
                              stopping_rounds=100,
                              leaderboard_frame = test_vorPandemie_h20,
                              nfolds=0,
                              max_runtime_secs=18*60*60,
                              #exclude_algos="GLM",
                              include_algos = c("GLM", "DeepLearning", "DRF","XGBoost","StackedEnsemble","GBM","DRF"),
                              weights_column="gewichtungs_modell")


save(aml_pandemie_14,file="/ModelleCovid/weight/aml_pandemie_14.RData")

#aml_pandemie <- miceadds::load.data(file="aml_pandemie_val.RData",type="RData")
#aml_test_pandemie <- miceadds::load.data(file="aml_test_pandemie.RData",type="RData")
#ll <- miceadds::load.data(file="aml_test_pandemie.RData",type="RData")


####
#  Leaderboard Test
####

lb_vorPandemie_h20 <- h2o.get_leaderboard(object = aml_pandemie_14, extra_columns = "ALL")
lb_vorPandemie_h20
save(lb_vorPandemie_h20,file="/ModelleCovid/weight/lb_vorPandemie_h20.RData")

###
#  Get Models
###

model <- as.vector(aml_pandemie_14@leaderboard$model_id)
model

###
#  Leaderboard in Panemie
###

# Alle Modelle vorhersagen











lb_vorPandemie_h20_df <- as.data.frame(lb_vorPandemie_h20)

lb_vorPandemie_h20_df_Stacked      <- lb_vorPandemie_h20_df %>% filter(algo=="StackedEnsemble")
lb_vorPandemie_h20_df_DeepLearning <- lb_vorPandemie_h20_df %>% filter(algo=="DeepLearning")
lb_vorPandemie_h20_df_DRF          <- lb_vorPandemie_h20_df %>% filter(algo=="DRF")
lb_vorPandemie_h20_df_GBM          <- lb_vorPandemie_h20_df %>% filter(algo=="GBM")
lb_vorPandemie_h20_df_GLM          <- lb_vorPandemie_h20_df %>% filter(algo=="GLM")
lb_vorPandemie_h20_df_XGBoost      <- lb_vorPandemie_h20_df %>% filter(algo=="XGBoost")

openxlsx::write.xlsx(rbind(lb_vorPandemie_h20_df[1,],
                           lb_vorPandemie_h20_df_Stacked[1,],     
                           lb_vorPandemie_h20_df_DeepLearning[1,],
                           lb_vorPandemie_h20_df_DRF[1,],         
                           lb_vorPandemie_h20_df_GBM[1,],         
                           lb_vorPandemie_h20_df_GLM[1,],         
                           lb_vorPandemie_h20_df_XGBoost[1,]     ),"/ModelleCovid/weight/besteAlgos14.xlsx")





###########################################################################################################################################################################
#################################                                   Leaderboard in pandemie                               #################################################
###########################################################################################################################################################################
pr_ci <- function(x,y){
  nu <- log(x$auc.integral/(1-x$auc.integral))
  tau <- 1/sqrt(y*x$auc.integral*(1-x$auc.integral))
  lower <- exp(nu-1.96*tau)/(1+exp(nu-1.96*tau))
  lower
  upper <- exp(nu+1.96*tau)/(1+exp(nu+1.96*tau))
  upper
  return(paste(round(x$auc.integral,3)," [",round(lower,3)," to ",round(upper,3),"]",sep=""))
  
}
###############################################################################################################################################################################




library(pROC)
library(PRROC)

predictions_vor_pandemie <- data.frame(truth=vorPandemie_validierung$verstorben)
predictions_in_pandemie <- data.frame(truth=pandemie$verstorben)
Leaderboard_Pandemie <- data.frame(modelname=NA,
                                   algo=NA,
                                   AUC_vor_Pandemie=NA,
                                   AUC_mitCI_vor_Pandemie=NA,
                                   PR_vor_Pandemie=NA,
                                   PR_mitCI_vor_Pandemie=NA,
                                   AUC_in_Pandemie=NA,
                                   AUC_mitCI_in_Pandemie=NA,
                                   PR_in_Pandemie=NA,
                                   PR_mitCI_in_Pandemie=NA)

for(i in 1:length(model))
{
  model_loop <-h2o.getModel(model[i])
  
  save(model_loop,file=paste("/ModelleCovid/weight/Modellsicherung/Mortality_covid_",model[i],"RData",sep=""))
  h2o.saveModel(object = model_loop, path = "/ModelleCovid/weight/Modellsicherung/", force = TRUE,filename=paste("Mortality_covid_",model[i],sep=""))
  # Predcit vor Pandemie
  
  p_t_vor_pandemie  <- h2o.predict(model_loop, newdata = valid_vorPandemie_h20)
  p_t_in_pandemie   <- h2o.predict(model_loop, newdata = pandemie_h20)
  
  # in Dataframe umwandeln
  
  p_t_vor_pandemie_df  <- as.data.frame(p_t_vor_pandemie  )
  p_t_in_pandemie_df   <- as.data.frame(p_t_in_pandemie   )
  
  
  
  
  predictions_vor_pandemie[,i+1] <- p_t_vor_pandemie_df$p1
  predictions_in_pandemie[,i+1]  <- p_t_in_pandemie_df$p1
  
  # AUC
  aaa_vor_pandemie <- pROC::ci.auc(predictions_vor_pandemie$truth,p_t_vor_pandemie_df$p1)
  aaa_in_pandemie  <- pROC::ci.auc(predictions_in_pandemie$truth,p_t_in_pandemie_df$p1)
  
  # PR
  
  pr_vor_pandemie <- pr.curve(scores.class0 = predictions_vor_pandemie[,i+1][predictions_vor_pandemie$truth==1],  predictions_vor_pandemie[,i+1][predictions_vor_pandemie$truth==0], curve = TRUE)
  pr_in_pandemie  <- pr.curve(scores.class0 = predictions_in_pandemie[,i+1][predictions_in_pandemie$truth==1],  predictions_in_pandemie[,i+1][predictions_in_pandemie$truth==0], curve = TRUE)
  
  
  pr_ci_vor_pandemie <- pr_ci(pr_vor_pandemie,length(predictions_vor_pandemie$truth))
  pr_ci_in_pandemie  <- pr_ci(pr_in_pandemie,length(predictions_in_pandemie$truth))
  
  Leaderboard_Pandemie <- rbind(Leaderboard_Pandemie,
                                c(lb_vorPandemie_h20[i,1][1],lb_vorPandemie_h20[i,10][1],aaa_vor_pandemie[2],paste(sprintf("%.3f",aaa_vor_pandemie[2])," [",sprintf("%.3f",aaa_vor_pandemie[1])," to ",sprintf("%.3f",aaa_vor_pandemie[3]),"]",sep=""),
                                  pr_vor_pandemie$auc.integral,pr_ci_vor_pandemie,
                                  aaa_in_pandemie[2],paste(sprintf("%.3f",aaa_in_pandemie[2])," [",sprintf("%.3f",aaa_in_pandemie[1])," to ",sprintf("%.3f",aaa_in_pandemie[3]),"]",sep=""),
                                  pr_in_pandemie$auc.integral,pr_ci_in_pandemie))
  
  
  
  
  print(paste(i," of ",length(model),sep=""))  
  
}


openxlsx::write.xlsx(Leaderboard_Pandemie,file="/ModelleCovid/weight/Leaderboard_Pandemie.xlsx")

Leaderboard_Pandemie <- Leaderboard_Pandemie %>% filter(is.na(modelname)==FALSE)

#Leaderboard_Pandemie <- openxlsx::read.xlsx("/ModelleCovid/weight/Leaderboard_Pandemie.xlsx")
#Leaderboard_Pandemie# Leaderboard_Pandemie
###########################################################################################################################################################################
#################################                             Raussuchen der jeweils besten Algos                         #################################################
###########################################################################################################################################################################
Leaderboard_Pandemie$AUC_vor_Pandemie <- as.numeric(Leaderboard_Pandemie$AUC_vor_Pandemie)
Leaderboard_Pandemie$PR_vor_Pandemie  <- as.numeric(Leaderboard_Pandemie$PR_vor_Pandemie)
Leaderboard_Pandemie$AUC_in_Pandemie  <- as.numeric(Leaderboard_Pandemie$AUC_in_Pandemie)
Leaderboard_Pandemie$PR_in_Pandemie   <- as.numeric(Leaderboard_Pandemie$PR_in_Pandemie)

Leaderboard_Pandemie <- Leaderboard_Pandemie %>% arrange(desc(AUC_vor_Pandemie)) %>% mutate(Reihenfolge.AUC.vor=1:n())
Leaderboard_Pandemie <- Leaderboard_Pandemie %>% arrange(desc(PR_vor_Pandemie)) %>% mutate(Reihenfolge.PR.vor=1:n())
Leaderboard_Pandemie <- Leaderboard_Pandemie %>% arrange(desc(AUC_in_Pandemie)) %>% mutate(Reihenfolge.AUC.in=1:n())
Leaderboard_Pandemie <- Leaderboard_Pandemie %>% arrange(desc(PR_in_Pandemie)) %>% mutate(Reihenfolge.PR.in=1:n())



lb_Stacked      <- Leaderboard_Pandemie %>% filter(algo=="StackedEnsemble")
lb_DeepLearning <- Leaderboard_Pandemie %>% filter(algo=="DeepLearning")
lb_DRF          <- Leaderboard_Pandemie %>% filter(algo=="DRF")
lb_GBM          <- Leaderboard_Pandemie %>% filter(algo=="GBM")
lb_GLM          <- Leaderboard_Pandemie %>% filter(algo=="GLM")
lb_XGBoost      <- Leaderboard_Pandemie %>% filter(algo=="XGBoost")




lb_Stacked      <- lb_Stacked     %>% arrange(desc(PR_vor_Pandemie))
lb_DeepLearning <- lb_DeepLearning%>% arrange(desc(PR_vor_Pandemie))
lb_DRF          <- lb_DRF         %>% arrange(desc(PR_vor_Pandemie))
lb_GBM          <- lb_GBM         %>% arrange(desc(PR_vor_Pandemie))
lb_GLM          <- lb_GLM         %>% arrange(desc(PR_vor_Pandemie))
lb_XGBoost      <- lb_XGBoost     %>% arrange(desc(PR_vor_Pandemie))

beste_vor_pandemie <- rbind(lb_Stacked     [1,],
                            lb_DeepLearning[1,],
                            lb_DRF         [1,],
                            lb_GBM         [1,],
                            lb_GLM         [1,],
                            lb_XGBoost     [1,])

beste_vor_pandemie %>% arrange(desc(PR_vor_Pandemie))


lb_Stacked      <- lb_Stacked     %>% arrange(desc(PR_in_Pandemie))
lb_DeepLearning <- lb_DeepLearning%>% arrange(desc(PR_in_Pandemie))
lb_DRF          <- lb_DRF         %>% arrange(desc(PR_in_Pandemie))
lb_GBM          <- lb_GBM         %>% arrange(desc(PR_in_Pandemie))
lb_GLM          <- lb_GLM         %>% arrange(desc(PR_in_Pandemie))
lb_XGBoost      <- lb_XGBoost     %>% arrange(desc(PR_in_Pandemie))

beste_in_pandemie <- rbind( lb_Stacked     [1,],
                            lb_DeepLearning[1,],
                            lb_DRF         [1,],
                            lb_GBM         [1,],
                            lb_GLM         [1,],
                            lb_XGBoost     [1,])

beste_in_pandemie %>% arrange(desc(PR_in_Pandemie))

openxlsx::write.xlsx(beste_vor_pandemie,"/ModelleCovid/weight/besteAlgos_vor_pandemie.xlsx")
openxlsx::write.xlsx(beste_in_pandemie,"/ModelleCovid/weight/besteAlgos_in_pandemie.xlsx")


###########################################################################################################################################################################
#################################                                           Kurven                                        #################################################
###########################################################################################################################################################################
####
#   PR mit besten vor Pandemie
####


best_vor_Pandemie_Stacked      <- lb_Stacked     %>% arrange(desc(PR_vor_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_vor_Pandemie_DeepLearning <- lb_DeepLearning%>% arrange(desc(PR_vor_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_vor_Pandemie_DRF          <- lb_DRF         %>% arrange(desc(PR_vor_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_vor_Pandemie_GBM          <- lb_GBM         %>% arrange(desc(PR_vor_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_vor_Pandemie_GLM          <- lb_GLM         %>% arrange(desc(PR_vor_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_vor_Pandemie_XGBoost      <- lb_XGBoost     %>% arrange(desc(PR_vor_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)


#StackedEnsemble_BestOfFamily_6_AutoML_1_20221205_81006 <- h2o.loadModel("/ModelleCovid/weight/Modellsicherung/Mortality_covid_StackedEnsemble_BestOfFamily_6_AutoML_1_20221205_81006")


pr_vor_Panemie_Stacked      <- data.frame(pr.curve(scores.class0 = predictions_vor_pandemie[,which(model==best_vor_Pandemie_Stacked     $modelname)+1][predictions_vor_pandemie$truth==1],  predictions_vor_pandemie[,which(model==best_vor_Pandemie_Stacked     $modelname)+1][predictions_vor_pandemie$truth==0], curve = TRUE)$curve)
pr_vor_Panemie_DeepLearning <- data.frame(pr.curve(scores.class0 = predictions_vor_pandemie[,which(model==best_vor_Pandemie_DeepLearning$modelname)+1][predictions_vor_pandemie$truth==1],  predictions_vor_pandemie[,which(model==best_vor_Pandemie_DeepLearning$modelname)+1][predictions_vor_pandemie$truth==0], curve = TRUE)$curve)
pr_vor_Panemie_DRF          <- data.frame(pr.curve(scores.class0 = predictions_vor_pandemie[,which(model==best_vor_Pandemie_DRF         $modelname)+1][predictions_vor_pandemie$truth==1],  predictions_vor_pandemie[,which(model==best_vor_Pandemie_DRF         $modelname)+1][predictions_vor_pandemie$truth==0], curve = TRUE)$curve)
pr_vor_Panemie_GBM          <- data.frame(pr.curve(scores.class0 = predictions_vor_pandemie[,which(model==best_vor_Pandemie_GBM         $modelname)+1][predictions_vor_pandemie$truth==1],  predictions_vor_pandemie[,which(model==best_vor_Pandemie_GBM         $modelname)+1][predictions_vor_pandemie$truth==0], curve = TRUE)$curve)
pr_vor_Panemie_GLM          <- data.frame(pr.curve(scores.class0 = predictions_vor_pandemie[,which(model==best_vor_Pandemie_GLM         $modelname)+1][predictions_vor_pandemie$truth==1],  predictions_vor_pandemie[,which(model==best_vor_Pandemie_GLM         $modelname)+1][predictions_vor_pandemie$truth==0], curve = TRUE)$curve)
pr_vor_Panemie_XGBoost      <- data.frame(pr.curve(scores.class0 = predictions_vor_pandemie[,which(model==best_vor_Pandemie_XGBoost     $modelname)+1][predictions_vor_pandemie$truth==1],  predictions_vor_pandemie[,which(model==best_vor_Pandemie_XGBoost     $modelname)+1][predictions_vor_pandemie$truth==0], curve = TRUE)$curve)



pr_vor_Panemie_Stacked     $dataset <- "Stacked Ensemble"
pr_vor_Panemie_DeepLearning$dataset <- "DeepLearning"
pr_vor_Panemie_DRF         $dataset <- "Deep random forest"
pr_vor_Panemie_GBM         $dataset <- "Gradient boosting machine"
pr_vor_Panemie_GLM         $dataset <- "Generalized linear model"
pr_vor_Panemie_XGBoost     $dataset <- "XGboost"





pr_curve_df <- rbind(pr_vor_Panemie_Stacked     ,
                     pr_vor_Panemie_DeepLearning,
                     pr_vor_Panemie_DRF         ,
                     pr_vor_Panemie_GBM         ,
                     pr_vor_Panemie_GLM         ,
                     pr_vor_Panemie_XGBoost     )

names(pr_curve_df)
names(pr_curve_df) <- c("Recall","Precision","threshold","dataset")

pr_curve_df <- pr_curve_df %>% arrange( dataset,desc(Precision),Recall)


ggplot(data=pr_curve_df ,aes(x=Recall,y=Precision,color=dataset))+
  geom_line(linewidth=2)+
  theme_bw()+
  geom_hline(yintercept = 0.01020408 ,linetype=2)+
  scale_y_continuous(lim=c(0,1),expand = c(0.01, 0.01))+
  scale_x_continuous(expand = c(0.01, 0.01)) +
  theme(axis.title = element_text(size=20),
        axis.text = element_text(size=18,color="black"),
        legend.text = element_text(size=18),
        legend.title = element_text(size=20),
        panel.grid = element_blank(),
        legend.position = c(.8,.8))+
  scale_color_manual("",values=c("#00BF7D","#00B4C5","#0073E6","#B3C7F7","#F57600","#B51963"))
graph2ppt(file="/ModelleCovid/weight/PowerPointGrafiken/PR beste Algorihmen vor Pandemie.pptx")


###
#  AUC
###

auc_vor_Panemie_Stacked      <- data.frame(sensitivities=pROC::roc(predictions_vor_pandemie$truth,predictions_vor_pandemie[,which(model==best_vor_Pandemie_Stacked     $modelname)+1])$sensitivities,eins_minus_specificities=1-pROC::roc(predictions_vor_pandemie$truth,predictions_vor_pandemie[,which(model==best_vor_Pandemie_Stacked     $modelname)+1])$specificities)
auc_vor_Panemie_DeepLearning <- data.frame(sensitivities=pROC::roc(predictions_vor_pandemie$truth,predictions_vor_pandemie[,which(model==best_vor_Pandemie_DeepLearning$modelname)+1])$sensitivities,eins_minus_specificities=1-pROC::roc(predictions_vor_pandemie$truth,predictions_vor_pandemie[,which(model==best_vor_Pandemie_DeepLearning$modelname)+1])$specificities)
auc_vor_Panemie_DRF          <- data.frame(sensitivities=pROC::roc(predictions_vor_pandemie$truth,predictions_vor_pandemie[,which(model==best_vor_Pandemie_DRF         $modelname)+1])$sensitivities,eins_minus_specificities=1-pROC::roc(predictions_vor_pandemie$truth,predictions_vor_pandemie[,which(model==best_vor_Pandemie_DRF         $modelname)+1])$specificities)
auc_vor_Panemie_GBM          <- data.frame(sensitivities=pROC::roc(predictions_vor_pandemie$truth,predictions_vor_pandemie[,which(model==best_vor_Pandemie_GBM         $modelname)+1])$sensitivities,eins_minus_specificities=1-pROC::roc(predictions_vor_pandemie$truth,predictions_vor_pandemie[,which(model==best_vor_Pandemie_GBM         $modelname)+1])$specificities)
auc_vor_Panemie_GLM          <- data.frame(sensitivities=pROC::roc(predictions_vor_pandemie$truth,predictions_vor_pandemie[,which(model==best_vor_Pandemie_GLM         $modelname)+1])$sensitivities,eins_minus_specificities=1-pROC::roc(predictions_vor_pandemie$truth,predictions_vor_pandemie[,which(model==best_vor_Pandemie_GLM         $modelname)+1])$specificities)
auc_vor_Panemie_XGBoost      <- data.frame(sensitivities=pROC::roc(predictions_vor_pandemie$truth,predictions_vor_pandemie[,which(model==best_vor_Pandemie_XGBoost     $modelname)+1])$sensitivities,eins_minus_specificities=1-pROC::roc(predictions_vor_pandemie$truth,predictions_vor_pandemie[,which(model==best_vor_Pandemie_XGBoost     $modelname)+1])$specificities)


auc_vor_Panemie_Stacked     $dataset <- "Stacked Ensemble"
auc_vor_Panemie_DeepLearning$dataset <- "DeepLearning"
auc_vor_Panemie_DRF         $dataset <- "Deep random forest"
auc_vor_Panemie_GBM         $dataset <- "Gradient boosting machine"
auc_vor_Panemie_GLM         $dataset <- "Generalized linear model"
auc_vor_Panemie_XGBoost     $dataset <- "XGboost"





auc_curve_vor_Pandemie_df <- rbind(auc_vor_Panemie_Stacked     ,
                                   auc_vor_Panemie_DeepLearning,
                                   auc_vor_Panemie_DRF         ,
                                   auc_vor_Panemie_GBM         ,
                                   auc_vor_Panemie_GLM         ,
                                   auc_vor_Panemie_XGBoost     )

names(auc_curve_vor_Pandemie_df)


pr_curve_df <- pr_curve_df %>% arrange( dataset,desc(Precision),Recall)


ggplot(data=auc_curve_vor_Pandemie_df ,aes(x=eins_minus_specificities,y=sensitivities,color=dataset))+
  geom_line(linewidth=2)+
  theme_bw()+
  #geom_hline(yintercept = 0.01020408 ,linetype=2)+
  scale_y_continuous(lim=c(0,1),expand = c(0.01, 0.01),"Sensitivity")+
  scale_x_continuous(expand = c(0.01, 0.01),"1-Specificity") +
  theme(axis.title = element_text(size=20),
        axis.text = element_text(size=18,color="black"),
        legend.text = element_text(size=18),
        legend.title = element_text(size=20),
        panel.grid = element_blank(),
        legend.position = c(.8,.5))+
  scale_color_manual("",values=c("#00BF7D","#00B4C5","#0073E6","#B3C7F7","#F57600","#B51963"))
graph2ppt(file="/ModelleCovid/weight/PowerPointGrafiken/AUC beste Algorihmen vor Pandemie.pptx")



####
#   PR mit besten in Pandemie
####



best_in_Pandemie_Stacked      <- lb_Stacked     %>% arrange(desc(PR_in_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_in_Pandemie_DeepLearning <- lb_DeepLearning%>% arrange(desc(PR_in_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_in_Pandemie_DRF          <- lb_DRF         %>% arrange(desc(PR_in_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_in_Pandemie_GBM          <- lb_GBM         %>% arrange(desc(PR_in_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_in_Pandemie_GLM          <- lb_GLM         %>% arrange(desc(PR_in_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_in_Pandemie_XGBoost      <- lb_XGBoost     %>% arrange(desc(PR_in_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)





pr_in_Panemie_Stacked      <- data.frame(pr.curve(scores.class0 = predictions_in_pandemie[,which(model==best_in_Pandemie_Stacked     $modelname)+1][predictions_in_pandemie$truth==1],  predictions_in_pandemie[,which(model==best_in_Pandemie_Stacked     $modelname)+1][predictions_in_pandemie$truth==0], curve = TRUE)$curve)
pr_in_Panemie_DeepLearning <- data.frame(pr.curve(scores.class0 = predictions_in_pandemie[,which(model==best_in_Pandemie_DeepLearning$modelname)+1][predictions_in_pandemie$truth==1],  predictions_in_pandemie[,which(model==best_in_Pandemie_DeepLearning$modelname)+1][predictions_in_pandemie$truth==0], curve = TRUE)$curve)
pr_in_Panemie_DRF          <- data.frame(pr.curve(scores.class0 = predictions_in_pandemie[,which(model==best_in_Pandemie_DRF         $modelname)+1][predictions_in_pandemie$truth==1],  predictions_in_pandemie[,which(model==best_in_Pandemie_DRF         $modelname)+1][predictions_in_pandemie$truth==0], curve = TRUE)$curve)
pr_in_Panemie_GBM          <- data.frame(pr.curve(scores.class0 = predictions_in_pandemie[,which(model==best_in_Pandemie_GBM         $modelname)+1][predictions_in_pandemie$truth==1],  predictions_in_pandemie[,which(model==best_in_Pandemie_GBM         $modelname)+1][predictions_in_pandemie$truth==0], curve = TRUE)$curve)
pr_in_Panemie_GLM          <- data.frame(pr.curve(scores.class0 = predictions_in_pandemie[,which(model==best_in_Pandemie_GLM         $modelname)+1][predictions_in_pandemie$truth==1],  predictions_in_pandemie[,which(model==best_in_Pandemie_GLM         $modelname)+1][predictions_in_pandemie$truth==0], curve = TRUE)$curve)
pr_in_Panemie_XGBoost      <- data.frame(pr.curve(scores.class0 = predictions_in_pandemie[,which(model==best_in_Pandemie_XGBoost     $modelname)+1][predictions_in_pandemie$truth==1],  predictions_in_pandemie[,which(model==best_in_Pandemie_XGBoost     $modelname)+1][predictions_in_pandemie$truth==0], curve = TRUE)$curve)



pr_in_Panemie_Stacked     $dataset <- "Stacked Ensemble"
pr_in_Panemie_DeepLearning$dataset <- "DeepLearning"
pr_in_Panemie_DRF         $dataset <- "Deep random forest"
pr_in_Panemie_GBM         $dataset <- "Gradient boosting machine"
pr_in_Panemie_GLM         $dataset <- "Generalized linear model"
pr_in_Panemie_XGBoost     $dataset <- "XGboost"





pr_curve_df <- rbind(pr_in_Panemie_Stacked     ,
                     pr_in_Panemie_DeepLearning,
                     pr_in_Panemie_DRF         ,
                     pr_in_Panemie_GBM         ,
                     pr_in_Panemie_GLM         ,
                     pr_in_Panemie_XGBoost     )

names(pr_curve_df)
names(pr_curve_df) <- c("Recall","Precision","threshold","dataset")

pr_curve_df <- pr_curve_df %>% arrange( dataset,desc(Precision),Recall)


ggplot(data=pr_curve_df ,aes(x=Recall,y=Precision,color=dataset))+
  geom_line(linewidth=2)+
  theme_bw()+
  geom_hline(yintercept = 0.01020408 ,linetype=2)+
  scale_y_continuous(lim=c(0,1),expand = c(0.01, 0.01))+
  scale_x_continuous(expand = c(0.01, 0.01)) +
  theme(axis.title = element_text(size=20),
        axis.text = element_text(size=18,color="black"),
        legend.text = element_text(size=18),
        legend.title = element_text(size=20),
        panel.grid = element_blank(),
        legend.position = c(.8,.8))+
  scale_color_manual("",values=c("#00BF7D","#00B4C5","#0073E6","#B3C7F7","#F57600","#B51963"))
graph2ppt(file="/ModelleCovid/weight/PowerPointGrafiken/PR beste Algorihmen in Pandemie.pptx")


###
#  AUC
###

auc_in_Panemie_Stacked      <- data.frame(sensitivities=pROC::roc(predictions_in_pandemie$truth,predictions_in_pandemie[,which(model==best_in_Pandemie_Stacked     $modelname)+1])$sensitivities,eins_minus_specificities=1-pROC::roc(predictions_in_pandemie$truth,predictions_in_pandemie[,which(model==best_in_Pandemie_Stacked     $modelname)+1])$specificities)
auc_in_Panemie_DeepLearning <- data.frame(sensitivities=pROC::roc(predictions_in_pandemie$truth,predictions_in_pandemie[,which(model==best_in_Pandemie_DeepLearning$modelname)+1])$sensitivities,eins_minus_specificities=1-pROC::roc(predictions_in_pandemie$truth,predictions_in_pandemie[,which(model==best_in_Pandemie_DeepLearning$modelname)+1])$specificities)
auc_in_Panemie_DRF          <- data.frame(sensitivities=pROC::roc(predictions_in_pandemie$truth,predictions_in_pandemie[,which(model==best_in_Pandemie_DRF         $modelname)+1])$sensitivities,eins_minus_specificities=1-pROC::roc(predictions_in_pandemie$truth,predictions_in_pandemie[,which(model==best_in_Pandemie_DRF         $modelname)+1])$specificities)
auc_in_Panemie_GBM          <- data.frame(sensitivities=pROC::roc(predictions_in_pandemie$truth,predictions_in_pandemie[,which(model==best_in_Pandemie_GBM         $modelname)+1])$sensitivities,eins_minus_specificities=1-pROC::roc(predictions_in_pandemie$truth,predictions_in_pandemie[,which(model==best_in_Pandemie_GBM         $modelname)+1])$specificities)
auc_in_Panemie_GLM          <- data.frame(sensitivities=pROC::roc(predictions_in_pandemie$truth,predictions_in_pandemie[,which(model==best_in_Pandemie_GLM         $modelname)+1])$sensitivities,eins_minus_specificities=1-pROC::roc(predictions_in_pandemie$truth,predictions_in_pandemie[,which(model==best_in_Pandemie_GLM         $modelname)+1])$specificities)
auc_in_Panemie_XGBoost      <- data.frame(sensitivities=pROC::roc(predictions_in_pandemie$truth,predictions_in_pandemie[,which(model==best_in_Pandemie_XGBoost     $modelname)+1])$sensitivities,eins_minus_specificities=1-pROC::roc(predictions_in_pandemie$truth,predictions_in_pandemie[,which(model==best_in_Pandemie_XGBoost     $modelname)+1])$specificities)


auc_in_Panemie_Stacked     $dataset <- "Stacked Ensemble"
auc_in_Panemie_DeepLearning$dataset <- "DeepLearning"
auc_in_Panemie_DRF         $dataset <- "Deep random forest"
auc_in_Panemie_GBM         $dataset <- "Gradient boosting machine"
auc_in_Panemie_GLM         $dataset <- "Generalized linear model"
auc_in_Panemie_XGBoost     $dataset <- "XGboost"





auc_curve_in_Pandemie_df <- rbind(auc_in_Panemie_Stacked     ,
                                  auc_in_Panemie_DeepLearning,
                                  auc_in_Panemie_DRF         ,
                                  auc_in_Panemie_GBM         ,
                                  auc_in_Panemie_GLM         ,
                                  auc_in_Panemie_XGBoost     )

names(auc_curve_in_Pandemie_df)


pr_curve_df <- pr_curve_df %>% arrange( dataset,desc(Precision),Recall)


ggplot(data=auc_curve_in_Pandemie_df ,aes(x=eins_minus_specificities,y=sensitivities,color=dataset))+
  geom_line(linewidth=2)+
  theme_bw()+
  #geom_hline(yintercept = 0.01020408 ,linetype=2)+
  scale_y_continuous(lim=c(0,1),expand = c(0.01, 0.01),"Sensitivity")+
  scale_x_continuous(expand = c(0.01, 0.01),"1-Specificity") +
  theme(axis.title = element_text(size=20),
        axis.text = element_text(size=18,color="black"),
        legend.text = element_text(size=18),
        legend.title = element_text(size=20),
        panel.grid = element_blank(),
        legend.position = c(.8,.5))+
  scale_color_manual("",values=c("#00BF7D","#00B4C5","#0073E6","#B3C7F7","#F57600","#B51963"))
graph2ppt(file="/ModelleCovid/weight/PowerPointGrafiken/AUC beste Algorihmen in Pandemie.pptx")



###########################################################################################################################################################################
#################################                                         Importance                                      #################################################
###########################################################################################################################################################################

best_vor_Pandemie_Stacked      <- lb_Stacked     %>% arrange(desc(PR_vor_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_vor_Pandemie_DeepLearning <- lb_DeepLearning%>% arrange(desc(PR_vor_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_vor_Pandemie_DRF          <- lb_DRF         %>% arrange(desc(PR_vor_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_vor_Pandemie_GBM          <- lb_GBM         %>% arrange(desc(PR_vor_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_vor_Pandemie_GLM          <- lb_GLM         %>% arrange(desc(PR_vor_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_vor_Pandemie_XGBoost      <- lb_XGBoost     %>% arrange(desc(PR_vor_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)



best_stacked      <- h2o.loadModel(paste("/ModelleCovid/weight/Modellsicherung/Mortality_covid_",best_vor_Pandemie_Stacked$modelname,sep=""))
best_DeepLearning <- h2o.loadModel(paste("/ModelleCovid/weight/Modellsicherung/Mortality_covid_",best_vor_Pandemie_DeepLearning$modelname,sep=""))
best_DR           <- h2o.loadModel(paste("/ModelleCovid/weight/Modellsicherung/Mortality_covid_",best_vor_Pandemie_DRF         $modelname,sep=""))
best_GBM          <- h2o.loadModel(paste("/ModelleCovid/weight/Modellsicherung/Mortality_covid_",best_vor_Pandemie_GBM         $modelname,sep=""))
best_GLM          <- h2o.loadModel(paste("/ModelleCovid/weight/Modellsicherung/Mortality_covid_",best_vor_Pandemie_GLM         $modelname,sep=""))
best_XGBoost      <- h2o.loadModel(paste("/ModelleCovid/weight/Modellsicherung/Mortality_covid_",best_vor_Pandemie_XGBoost     $modelname,sep=""))





vi_vor_Pandemie_Stacked     <- h2o.permutation_importance(best_stacked, newdata = train_vorPandemie_h20)
vi_vor_Pandemie_DeepLearning<- h2o.varimp(h2o.getModel(best_vor_Pandemie_DeepLearning$modelname))
vi_vor_Pandemie_DRF         <- h2o.varimp(h2o.getModel(best_vor_Pandemie_DRF         $modelname))
vi_vor_Pandemie_GBM         <- h2o.varimp(h2o.getModel(best_vor_Pandemie_GBM         $modelname))
vi_vor_Pandemie_GLM         <- h2o.varimp(h2o.getModel(best_vor_Pandemie_GLM         $modelname))
vi_vor_Pandemie_XGBoost     <- h2o.varimp(h2o.getModel(best_vor_Pandemie_XGBoost     $modelname))


write.csv2(vi_vor_Pandemie_Stacked     ,file="/ModelleCovid/weight/vi_vor_Pandemie_Stacked.csv",sep=",",dec=";")
write.csv2(vi_vor_Pandemie_DeepLearning,file="/ModelleCovid/weight/vi_vor_Pandemie_DeepLearning.csv",sep=",",dec=";")
write.csv2(vi_vor_Pandemie_DRF         ,file="/ModelleCovid/weight/vi_vor_Pandemie_DRF.csv",sep=",",dec=";")
write.csv2(vi_vor_Pandemie_GBM         ,file="/ModelleCovid/weight/vi_vor_Pandemie_GBM.csv",sep=",",dec=";")
write.csv2(vi_vor_Pandemie_GLM         ,file="/ModelleCovid/weight/vi_vor_Pandemie_GLM.csv",sep=",",dec=";")
write.csv2(vi_vor_Pandemie_XGBoost     ,file="/ModelleCovid/weight/vi_vor_Pandemie_XGBoost.csv",sep=",",dec=";")



vi_vor_pandemie <- list(Stacked_Ensemble=vi_vor_Pandemie_Stacked $Variable[1:100]    ,
                        DeepLearning=vi_vor_Pandemie_DeepLearning$variable[1:100],
                        DRF=vi_vor_Pandemie_DRF$variable[1:100]        ,
                        GBM=vi_vor_Pandemie_GBM$variable[1:100]        ,
                        GLM=vi_vor_Pandemie_GLM$variable[1:100]         ,
                        XGboost=vi_vor_Pandemie_XGBoost$variable[1:100]     )


vd <- ggVennDiagram(vi_vor_pandemie)
ggVennDiagram(vi_vor_pandemie,show_intersect = F,label_color = "white",label="count")
graph2ppt(file="/ModelleCovid/weight/PowerPointGrafiken/Venn_Pandemie.pptx")

####
# Extrahieren der Übereinstimmungen


Intersect <- function (x) {  
  # Multiple set version of intersect
  # x is a list
  if (length(x) == 1) {
    unlist(x)
  } else if (length(x) == 2) {
    intersect(x[[1]], x[[2]])
  } else if (length(x) > 2){
    intersect(x[[1]], Intersect(x[-1]))
  }
}

Union <- function (x) {  
  # Multiple set version of union
  # x is a list
  if (length(x) == 1) {
    unlist(x)
  } else if (length(x) == 2) {
    union(x[[1]], x[[2]])
  } else if (length(x) > 2) {
    union(x[[1]], Union(x[-1]))
  }
}

Setdiff <- function (x, y) {
  # Remove the union of the y's from the common x's. 
  # x and y are lists of characters.
  xx <- Intersect(x)
  yy <- Union(y)
  setdiff(xx, yy)
}

combs <- 
  unlist(lapply(1:length(vi_vor_pandemie), 
                function(j) combn(names(vi_vor_pandemie), j, simplify = FALSE)),
         recursive = FALSE)
names(combs) <- sapply(combs, function(i) paste0(i, collapse = ""))


elements <- 
  lapply(combs, function(i) Setdiff(vi_vor_pandemie[i], vi_vor_pandemie[setdiff(names(vi_vor_pandemie), i)]))

n.elements <- sapply(elements, length)
print(n.elements)

cbind.fill<-function(...){
  nm <- list(...) 
  nm<-lapply(nm, as.matrix)
  n <- max(sapply(nm, nrow)) 
  do.call(cbind, lapply(nm, function (x) 
    rbind(x, matrix(, n-nrow(x), ncol(x))))) 
}
df <-  as.data.frame(do.call(cbind.fill, elements))

names(df) <- names(elements)

openxlsx::write.xlsx(df,"/ModelleCovid/weight/Übereinstimmungen Venn Diagramm.xlsx")


str(vd)

graph2ppt(file="/ModelleCovid/weight/VennDiagramm Top100.pptx")

openxlsx::write.xlsx(Reduce(intersect, vi_vor_pandemie),"/ModelleCovid/weight/Intersect Variablen aller Modelle.xlsx")


###########################################################################################################################################################################
#################################                                       Plot Veränderung                                  #################################################
###########################################################################################################################################################################

library(dplyr)
library(ggplot2)
library(ggridges)
library(export)


head(Leaderboard_Pandemie)


Leaderboard_Pandemie_vor <- Leaderboard_Pandemie %>% select(modelname,algo,AUC_vor_Pandemie,PR_vor_Pandemie) %>% rename(AUC=AUC_vor_Pandemie,PR=PR_vor_Pandemie)
Leaderboard_Pandemie_in  <- Leaderboard_Pandemie %>% select(modelname,algo,AUC_in_Pandemie,PR_in_Pandemie)   %>% rename(AUC=AUC_in_Pandemie,PR=PR_in_Pandemie)

Leaderboard_Pandemie_vor$Zeitpunkt <- "vor"
Leaderboard_Pandemie_in $Zeitpunkt <- "in"


Leaderboard_Pandemie_long <- rbind(Leaderboard_Pandemie_vor,
                                   Leaderboard_Pandemie_in )
Leaderboard_Pandemie_long <- na.omit(Leaderboard_Pandemie_long)

Leaderboard_Pandemie_long$Zeitpunkt <- factor(Leaderboard_Pandemie_long$Zeitpunkt,levels  =c("vor","in"))
Leaderboard_Pandemie_long$AUC <- as.numeric(as.character(Leaderboard_Pandemie_long$AUC))
Leaderboard_Pandemie_long$PR <- as.numeric(as.character(Leaderboard_Pandemie_long$PR))



ggplot(data=Leaderboard_Pandemie_long,aes(x=Zeitpunkt,y=AUC,color=modelname))+geom_line(aes(group=modelname))+geom_point(size=4,shape=21,stroke=2,fill="white") + theme_bw() + theme(legend.position = "none",panel.grid = element_blank(),axis.text = element_text(size=48,color="black"),axis.title=element_text(size=54))+scale_x_discrete("",labels=c("Three month before first wave","In first wave"))+scale_y_continuous("AUROC")
ggsave("/ModelleCovid/weight/AUROC Veränderung.jpg")
graph2ppt(file="/ModelleCovid/weight/AUROC Veränderung.pptx",margins = c(top = 0, right = 0, bottom = 0, left = 0),scaling = 200,aspectr=1)

ggplot(data=Leaderboard_Pandemie_long,aes(x=Zeitpunkt,y=PR,color=modelname)) +geom_line(aes(group=modelname))+geom_point(size=4,shape=21,stroke=2,fill="white") + theme_bw() + theme(legend.position = "none",panel.grid = element_blank(),axis.text = element_text(size=16,color="black"),axis.title=element_text(size=18))+scale_x_discrete("",labels=c("Three month before first wave","In first wave"))+scale_y_continuous("AUPR")
ggsave("/ModelleCovid/weight/AUPR Veränderung.jpg")
graph2ppt(file="/ModelleCovid/weight/AUPR Veränderung.pptx",margins = c(top = 0, right = 0, bottom = 0, left = 0),scaling = 200,aspectr=1)


ggplot(data=Leaderboard_Pandemie_long,aes(x=Zeitpunkt,y=AUC,color=modelname))+geom_line(aes(group=modelname))+geom_point(size=4,shape=21,stroke=2,fill="white") + theme_bw() +facet_grid(~algo) + theme(legend.position = "none",panel.grid = element_blank(),axis.text = element_text(size=16,color="black"),axis.title=element_text(size=18),strip.background = element_rect(fill="white"),strip.text = element_text(size=18))+scale_x_discrete("",labels=c("Three month before first wave","In first wave"))+scale_y_continuous("AUROC")
ggsave("/ModelleCovid/weight/AUROC Veränderung aufgeteilt nach Modellen.jpg")
graph2ppt(file="/ModelleCovid/weight/AUROC Veränderung aufgeteilt nach Modellen.pptx",margins = c(top = 0, right = 0, bottom = 0, left = 0),scaling = 200,aspectr=1)

ggplot(data=Leaderboard_Pandemie_long,aes(x=Zeitpunkt,y=PR,color=modelname)) +geom_line(aes(group=modelname))+geom_point(size=4,shape=21,stroke=2,fill="white") + theme_bw() +facet_grid(~algo) + theme(legend.position = "none",panel.grid = element_blank(),axis.text = element_text(size=16,color="black"),axis.title=element_text(size=18),strip.background = element_rect(fill="white"),strip.text = element_text(size=18))+scale_x_discrete("",labels=c("Three month before first wave","In first wave"))+scale_y_continuous("AUPR")
ggsave("/ModelleCovid/weight/AUPR Veränderung aufgeteilt nach Modellen.jpg")
graph2ppt(file="/ModelleCovid/weight/AUPR Veränderung aufgeteilt nach Modellen.pptx",margins = c(top = 0, right = 0, bottom = 0, left = 0),scaling = 200,aspectr=1)






