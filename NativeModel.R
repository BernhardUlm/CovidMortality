library(dplyr)
library(ggplot2)
library(h2o)

install.packages("devtools")
library(devtools)
devtools::install_github("tomwenseleers/export")
library(export)

devtools::install_github("gaospecial/ggVennDiagram")
library(ggVennDiagram)


h2o.init(min_mem_size = "850G")

# Load Data
train_vorPandemie           <- miceadds::load.data(path="RData/",file="Mortality Covid AutoMtrain_vorPandemie.RData",type="RData")
test_vorPandemie            <- miceadds::load.data(path="Data/",file="Mortality Covid AutoMtest_vorPandemie.RData",type="RData")
vorPandemie_validierung     <- miceadds::load.data(path="RData/",file="Mortality Covid AutoMvalid_vorPandemie.RData",type="RData")
pandemie                    <- miceadds::load.data(path="RData/",file="Mortality Covid AutoMpandemie.RData",type="RData")



# Create H2O Dataframes
train_vorPandemie_h20 <- as.h2o(train_vorPandemie)
test_vorPandemie_h20  <- as.h2o(test_vorPandemie)
valid_vorPandemie_h20 <- as.h2o(vorPandemie_validierung)
pandemie_h20          <- as.h2o(pandemie)


# Identify predictors and response
y <- "verstorben"
x <- setdiff(names(train_vorPandemie_h20), y)

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
                              max_runtime_secs=65*60*60,
                              #exclude_algos="GLM",
                              include_algos = c("GLM", "DeepLearning", "DRF","XGBoost","StackedEnsemble","GBM","DRF"))


save(aml_pandemie_14,file="ModelleCovid/aml_pandemie_14.RData")

#aml_pandemie <- miceadds::load.data(file="aml_pandemie_val.RData",type="RData")
#aml_test_pandemie <- miceadds::load.data(file="aml_test_pandemie.RData",type="RData")
#ll <- miceadds::load.data(file="aml_test_pandemie.RData",type="RData")


####
#  Leaderboard Test
####

lb_vorPandemie_h20 <- h2o.get_leaderboard(object = aml_pandemie_14, extra_columns = "ALL")
lb_vorPandemie_h20
save(lb_vorPandemie_h20,file="ModelleCovid/lb_vorPandemie_h20.RData")

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
                           lb_vorPandemie_h20_df_XGBoost[1,]     ),"ModelleCovid/besteAlgos14.xlsx")





###########################################################################################################################################################################
#################################                                   Leaderboard in pandemie                               #################################################
###########################################################################################################################################################################
# CI for PR
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
  
  save(model_loop,file=paste("ModelleCovid/Modellsicherung/Mortality_covid_",model[i],"RData",sep=""))
  h2o.saveModel(object = model_loop, path = "ModelleCovid/Modellsicherung/", force = TRUE,filename=paste("Mortality_covid_",model[i],sep=""))
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


openxlsx::write.xlsx(Leaderboard_Pandemie,file="ModelleCovid/Leaderboard_Pandemie.xlsx")

Leaderboard_Pandemie <- Leaderboard_Pandemie %>% filter(is.na(modelname)==FALSE)

#Leaderboard_Pandemie <- openxlsx::read.xlsx("ModelleCovid/Leaderboard_Pandemie.xlsx")
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

openxlsx::write.xlsx(beste_vor_pandemie,".../besteAlgos_vor_pandemie.xlsx")
openxlsx::write.xlsx(beste_in_pandemie,".../besteAlgos_in_pandemie.xlsx")


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



model_best_vor_Pandemie_Stacked     <- h2o.loadModel("Mortality_covid_StackedEnsemble_Best1000_1_AutoML_1_20230616_101009")
model_best_vor_Pandemie_DeepLearning<- h2o.loadModel("Mortality_covid_DeepLearning_grid_3_AutoML_1_20230616_101009_model_9")
model_best_vor_Pandemie_DRF         <- h2o.loadModel("Mortality_covid_DRF_1_AutoML_1_20230616_101009")
model_best_vor_Pandemie_GBM         <- h2o.loadModel("Mortality_covid_GBM_grid_1_AutoML_1_20230616_101009_model_27")
model_best_vor_Pandemie_GLM         <- h2o.loadModel("Mortality_covid_GLM_1_AutoML_1_20230616_101009")
model_best_vor_Pandemie_XGBoost     <- h2o.loadModel("Mortality_covid_XGBoost_grid_1_AutoML_1_20230616_101009_model_70")



p_t_vor_Pandemie_Stacked       <- h2o.predict(model_best_vor_Pandemie_Stacked     , newdata = valid_vorPandemie_h20)
p_t_vor_Pandemie_DeepLearning  <- h2o.predict(model_best_vor_Pandemie_DeepLearning, newdata = valid_vorPandemie_h20)
p_t_vor_Pandemie_DRF           <- h2o.predict(model_best_vor_Pandemie_DRF         , newdata = valid_vorPandemie_h20)
p_t_vor_Pandemie_GBM           <- h2o.predict(model_best_vor_Pandemie_GBM         , newdata = valid_vorPandemie_h20)
p_t_vor_Pandemie_GLM           <- h2o.predict(model_best_vor_Pandemie_GLM         , newdata = valid_vorPandemie_h20)
p_t_vor_Pandemie_XGBoost       <- h2o.predict(model_best_vor_Pandemie_XGBoost     , newdata = valid_vorPandemie_h20)



p_t_vor_pandemie_df  <- as.data.frame(p_t_vor_pandemie  )
p_t_in_pandemie_df   <- as.data.frame(p_t_in_pandemie   )

p_t_vor_Pandemie_Stacked_df      <- as.data.frame(p_t_vor_Pandemie_Stacked      )
p_t_vor_Pandemie_DeepLearning_df <- as.data.frame(p_t_vor_Pandemie_DeepLearning )
p_t_vor_Pandemie_DRF_df          <- as.data.frame(p_t_vor_Pandemie_DRF          )
p_t_vor_Pandemie_GBM_df          <- as.data.frame(p_t_vor_Pandemie_GBM          )
p_t_vor_Pandemie_GLM_df          <- as.data.frame(p_t_vor_Pandemie_GLM          )
p_t_vor_Pandemie_XGBoost_df      <- as.data.frame(p_t_vor_Pandemie_XGBoost      )

predictions_vor_pandemie <- data.frame(truth=vorPandemie_validierung$verstorben)

predictions_vor_pandemie$Stacked     <- p_t_vor_Pandemie_Stacked_df     $p1
predictions_vor_pandemie$DeepLearning<- p_t_vor_Pandemie_DeepLearning_df$p1
predictions_vor_pandemie$DRF         <- p_t_vor_Pandemie_DRF_df         $p1
predictions_vor_pandemie$GBM         <- p_t_vor_Pandemie_GBM_df         $p1
predictions_vor_pandemie$GLM         <- p_t_vor_Pandemie_GLM_df         $p1
predictions_vor_pandemie$XGBoost     <- p_t_vor_Pandemie_XGBoost_df     $p1

save(predictions_vor_pandemie,file="predictions_vor_pandemie.RData")

pr_vor_Panemie_Stacked      <- data.frame(pr.curve(scores.class0 = predictions_vor_pandemie$Stacked     [predictions_vor_pandemie$truth==1],  predictions_vor_pandemie$Stacked     [predictions_vor_pandemie$truth==0], curve = TRUE)$curve)
pr_vor_Panemie_DeepLearning <- data.frame(pr.curve(scores.class0 = predictions_vor_pandemie$DeepLearning[predictions_vor_pandemie$truth==1],  predictions_vor_pandemie$DeepLearning[predictions_vor_pandemie$truth==0], curve = TRUE)$curve)
pr_vor_Panemie_DRF          <- data.frame(pr.curve(scores.class0 = predictions_vor_pandemie$DRF         [predictions_vor_pandemie$truth==1],  predictions_vor_pandemie$DRF         [predictions_vor_pandemie$truth==0], curve = TRUE)$curve)
pr_vor_Panemie_GBM          <- data.frame(pr.curve(scores.class0 = predictions_vor_pandemie$GBM         [predictions_vor_pandemie$truth==1],  predictions_vor_pandemie$GBM         [predictions_vor_pandemie$truth==0], curve = TRUE)$curve)
pr_vor_Panemie_GLM          <- data.frame(pr.curve(scores.class0 = predictions_vor_pandemie$GLM         [predictions_vor_pandemie$truth==1],  predictions_vor_pandemie$GLM         [predictions_vor_pandemie$truth==0], curve = TRUE)$curve)
pr_vor_Panemie_XGBoost      <- data.frame(pr.curve(scores.class0 = predictions_vor_pandemie$XGBoost     [predictions_vor_pandemie$truth==1],  predictions_vor_pandemie$XGBoost     [predictions_vor_pandemie$truth==0], curve = TRUE)$curve)



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
graph2ppt(file="PowerPointGrafiken/PR beste Algorihmen vor Pandemie.pptx")


###
#  AUC
###

auc_vor_Panemie_Stacked      <- data.frame(sensitivities=pROC::roc(predictions_vor_pandemie$truth,predictions_vor_pandemie$Stacked     )$sensitivities,eins_minus_specificities=1-pROC::roc(predictions_vor_pandemie$truth,predictions_vor_pandemie$Stacked     )$specificities)
auc_vor_Panemie_DeepLearning <- data.frame(sensitivities=pROC::roc(predictions_vor_pandemie$truth,predictions_vor_pandemie$DeepLearning)$sensitivities,eins_minus_specificities=1-pROC::roc(predictions_vor_pandemie$truth,predictions_vor_pandemie$DeepLearning)$specificities)
auc_vor_Panemie_DRF          <- data.frame(sensitivities=pROC::roc(predictions_vor_pandemie$truth,predictions_vor_pandemie$DRF         )$sensitivities,eins_minus_specificities=1-pROC::roc(predictions_vor_pandemie$truth,predictions_vor_pandemie$DRF         )$specificities)
auc_vor_Panemie_GBM          <- data.frame(sensitivities=pROC::roc(predictions_vor_pandemie$truth,predictions_vor_pandemie$GBM         )$sensitivities,eins_minus_specificities=1-pROC::roc(predictions_vor_pandemie$truth,predictions_vor_pandemie$GBM         )$specificities)
auc_vor_Panemie_GLM          <- data.frame(sensitivities=pROC::roc(predictions_vor_pandemie$truth,predictions_vor_pandemie$GLM         )$sensitivities,eins_minus_specificities=1-pROC::roc(predictions_vor_pandemie$truth,predictions_vor_pandemie$GLM         )$specificities)
auc_vor_Panemie_XGBoost      <- data.frame(sensitivities=pROC::roc(predictions_vor_pandemie$truth,predictions_vor_pandemie$XGBoost     )$sensitivities,eins_minus_specificities=1-pROC::roc(predictions_vor_pandemie$truth,predictions_vor_pandemie$XGBoost     )$specificities)


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
graph2ppt(file="ModelleCovid/PowerPointGrafiken/AUC beste Algorihmen vor Pandemie.pptx")



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
graph2ppt(file="ModelleCovid/PowerPointGrafiken/PR beste Algorihmen in Pandemie.pptx")


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
graph2ppt(file="ModelleCovid/PowerPointGrafiken/AUC beste Algorihmen in Pandemie.pptx")



###########################################################################################################################################################################
#################################                                         Importance                                      #################################################
###########################################################################################################################################################################

best_vor_Pandemie_Stacked      <- lb_Stacked     %>% arrange(desc(PR_vor_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_vor_Pandemie_DeepLearning <- lb_DeepLearning%>% arrange(desc(PR_vor_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_vor_Pandemie_DRF          <- lb_DRF         %>% arrange(desc(PR_vor_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_vor_Pandemie_GBM          <- lb_GBM         %>% arrange(desc(PR_vor_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_vor_Pandemie_GLM          <- lb_GLM         %>% arrange(desc(PR_vor_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_vor_Pandemie_XGBoost      <- lb_XGBoost     %>% arrange(desc(PR_vor_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)



best_stacked      <- h2o.loadModel(paste("ModelleCovid/Modellsicherung/Mortality_covid_",best_vor_Pandemie_Stacked$modelname,sep=""))
best_DeepLearning <- h2o.loadModel(paste("ModelleCovid/Modellsicherung/Mortality_covid_",best_vor_Pandemie_DeepLearning$modelname,sep=""))
best_DR           <- h2o.loadModel(paste("ModelleCovid/Modellsicherung/Mortality_covid_",best_vor_Pandemie_DRF         $modelname,sep=""))
best_GBM          <- h2o.loadModel(paste("ModelleCovid/Modellsicherung/Mortality_covid_",best_vor_Pandemie_GBM         $modelname,sep=""))
best_GLM          <- h2o.loadModel(paste("ModelleCovid/Modellsicherung/Mortality_covid_",best_vor_Pandemie_GLM         $modelname,sep=""))
best_XGBoost      <- h2o.loadModel(paste("ModelleCovid/Modellsicherung/Mortality_covid_",best_vor_Pandemie_XGBoost     $modelname,sep=""))





vi_vor_Pandemie_Stacked     <- h2o.permutation_importance(best_stacked, newdata = train_vorPandemie_h20)
vi_vor_Pandemie_DeepLearning<- h2o.varimp(h2o.getModel(best_vor_Pandemie_DeepLearning$modelname))
vi_vor_Pandemie_DRF         <- h2o.varimp(h2o.getModel(best_vor_Pandemie_DRF         $modelname))
vi_vor_Pandemie_GBM         <- h2o.varimp(h2o.getModel(best_vor_Pandemie_GBM         $modelname))
vi_vor_Pandemie_GLM         <- h2o.varimp(h2o.getModel(best_vor_Pandemie_GLM         $modelname))
vi_vor_Pandemie_XGBoost     <- h2o.varimp(h2o.getModel(best_vor_Pandemie_XGBoost     $modelname))


write.csv2(vi_vor_Pandemie_Stacked     ,file="ModelleCovid/vi_vor_Pandemie_Stacked.csv",sep=",",dec=";")
write.csv2(vi_vor_Pandemie_DeepLearning,file="ModelleCovid/vi_vor_Pandemie_DeepLearning.csv",sep=",",dec=";")
write.csv2(vi_vor_Pandemie_DRF         ,file="ModelleCovid/vi_vor_Pandemie_DRF.csv",sep=",",dec=";")
write.csv2(vi_vor_Pandemie_GBM         ,file="ModelleCovid/vi_vor_Pandemie_GBM.csv",sep=",",dec=";")
write.csv2(vi_vor_Pandemie_GLM         ,file="ModelleCovid/vi_vor_Pandemie_GLM.csv",sep=",",dec=";")
write.csv2(vi_vor_Pandemie_XGBoost     ,file="ModelleCovid/vi_vor_Pandemie_XGBoost.csv",sep=",",dec=";")

# vi_vor_Pandemie_Stacked     <- read.csv(file="ModelleCovid/vi_vor_Pandemie_Stacked.csv",sep=";",dec=",")
# vi_vor_Pandemie_DeepLearning<- read.csv(file="ModelleCovid/vi_vor_Pandemie_DeepLearning.csv",sep=";",dec=",")
# vi_vor_Pandemie_DRF         <- read.csv(file="ModelleCovid/vi_vor_Pandemie_DRF.csv",sep=";",dec=",")
# vi_vor_Pandemie_GBM         <- read.csv(file="ModelleCovid/vi_vor_Pandemie_GBM.csv",sep=";",dec=",")
# vi_vor_Pandemie_GLM         <- read.csv(file="ModelleCovid/vi_vor_Pandemie_GLM.csv",sep=";",dec=",")
# vi_vor_Pandemie_XGBoost     <- read.csv(file="ModelleCovid/vi_vor_Pandemie_XGBoost.csv",sep=";",dec=",")


vi_vor_pandemie <- list(Stacked_Ensemble=vi_vor_Pandemie_Stacked $Variable[1:100]    ,
                        DeepLearning=vi_vor_Pandemie_DeepLearning$variable[1:100],
                        DRF=vi_vor_Pandemie_DRF$variable[1:100]        ,
                        GBM=vi_vor_Pandemie_GBM$variable[1:100]        ,
                        GLM=vi_vor_Pandemie_GLM$variable[1:100]         ,
                        XGboost=vi_vor_Pandemie_XGBoost$variable[1:100]     )


vd <- ggVennDiagram(vi_vor_pandemie)
ggVennDiagram(vi_vor_pandemie,show_intersect = F,label_color = "white",label="count")


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

openxlsx::write.xlsx(df,"ModelleCovid/Übereinstimmungen Venn Diagramm.xlsx")


str(vd)

graph2ppt(file="ModelleCovid/VennDiagramm Top100.pptx")

openxlsx::write.xlsx(Reduce(intersect, vi_vor_pandemie),"ModelleCovid/Intersect Variablen aller Modelle.xlsx")


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


ggplot(data=Leaderboard_Pandemie_long,aes(x=Zeitpunkt,y=AUC,color=modelname))+geom_line(aes(group=modelname))+geom_point(size=4,shape=21,stroke=2,fill="white") + theme_bw() + theme(legend.position = "none",panel.grid = element_blank(),axis.text = element_text(size=48,color="black"),axis.title=element_text(size=54))+scale_x_discrete("",labels=c("Three month before first wave","In first wave"))+scale_y_continuous("AUROC")
ggsave("AUROC Veränderung.jpg")
graph2ppt(file="ModelleCovid/AUROC Veränderung.pptx",margins = c(top = 0, right = 0, bottom = 0, left = 0),scaling = 200,aspectr=1)

ggplot(data=Leaderboard_Pandemie_long,aes(x=Zeitpunkt,y=PR,color=modelname)) +geom_line(aes(group=modelname))+geom_point(size=4,shape=21,stroke=2,fill="white") + theme_bw() + theme(legend.position = "none",panel.grid = element_blank(),axis.text = element_text(size=16,color="black"),axis.title=element_text(size=18))+scale_x_discrete("",labels=c("Three month before first wave","In first wave"))+scale_y_continuous("AUPR")
ggsave(".../AUPR Veränderung.jpg")
graph2ppt(file="ModelleCovid/AUPR Veränderung.pptx",margins = c(top = 0, right = 0, bottom = 0, left = 0),scaling = 200,aspectr=1)


ggplot(data=Leaderboard_Pandemie_long,aes(x=Zeitpunkt,y=AUC,color=modelname))+geom_line(aes(group=modelname))+geom_point(size=4,shape=21,stroke=2,fill="white") + theme_bw() +facet_grid(~algo) + theme(legend.position = "none",panel.grid = element_blank(),axis.text = element_text(size=16,color="black"),axis.title=element_text(size=18),strip.background = element_rect(fill="white"),strip.text = element_text(size=18))+scale_x_discrete("",labels=c("Three month before first wave","In first wave"))+scale_y_continuous("AUROC")
ggsave(".../AUROC Veränderung aufgeteilt nach Modellen.jpg")
graph2ppt(file="ModelleCovid/AUROC Veränderung aufgeteilt nach Modellen.pptx",margins = c(top = 0, right = 0, bottom = 0, left = 0),scaling = 200,aspectr=1)

ggplot(data=Leaderboard_Pandemie_long,aes(x=Zeitpunkt,y=PR,color=modelname)) +geom_line(aes(group=modelname))+geom_point(size=4,shape=21,stroke=2,fill="white") + theme_bw() +facet_grid(~algo) + theme(legend.position = "none",panel.grid = element_blank(),axis.text = element_text(size=16,color="black"),axis.title=element_text(size=18),strip.background = element_rect(fill="white"),strip.text = element_text(size=18))+scale_x_discrete("",labels=c("Three month before first wave","In first wave"))+scale_y_continuous("AUPR")
ggsave(".../AUPR Veränderung aufgeteilt nach Modellen.jpg")
graph2ppt(file="ModelleCovid/AUPR Veränderung aufgeteilt nach Modellen.pptx",margins = c(top = 0, right = 0, bottom = 0, left = 0),scaling = 200,aspectr=1)




################################################################################################################
#####
#  Für AUC und PR predict
##
pPandemie_model_GBM            <- h2o.predict(model_GBM         , newdata = pandemie_h20)
pPandemie_model_stacked        <- h2o.predict(model_stacked     , newdata = pandemie_h20)
pPandemie_model_XGBoost        <- h2o.predict(model_XGBoost     , newdata = pandemie_h20)
pPandemie_model_GLM            <- h2o.predict(model_GLM         , newdata = pandemie_h20)
pPandemie_model_DeepLearning   <- h2o.predict(model_DeepLearning, newdata = pandemie_h20)
pPandemie_model_DRF            <- h2o.predict(model_DRF         , newdata = pandemie_h20)

pPandemie_model_GBM_df         <- as.data.frame(pPandemie_model_GBM         )
pPandemie_model_stacked_df     <- as.data.frame(pPandemie_model_stacked     )
pPandemie_model_XGBoost_df     <- as.data.frame(pPandemie_model_XGBoost     )
pPandemie_model_GLM_df         <- as.data.frame(pPandemie_model_GLM         )
pPandemie_model_DeepLearning_df<- as.data.frame(pPandemie_model_DeepLearning)
pPandemie_model_DRF_df         <- as.data.frame(pPandemie_model_DRF         )

pPandemie_model_GBM_df         $truth <- pandemie$verstorben
pPandemie_model_stacked_df     $truth <- pandemie$verstorben
pPandemie_model_XGBoost_df     $truth <- pandemie$verstorben
pPandemie_model_GLM_df         $truth <- pandemie$verstorben
pPandemie_model_DeepLearning_df$truth <- pandemie$verstorben
pPandemie_model_DRF_df         $truth <- pandemie$verstorben


save(pPandemie_model_GBM_df          ,file="pPandemie_model_GBM_df.RData")    
save(pPandemie_model_stacked_df      ,file="pPandemie_model_stacked_df.RData")   
save(pPandemie_model_XGBoost_df      ,file="pPandemie_model_XGBoost_df.RData")  
save(pPandemie_model_GLM_df          ,file="pPandemie_model_GLM_df.RData") 
save(pPandemie_model_DeepLearning_df ,file="pPandemie_model_DeepLearning_df.RData")
save(pPandemie_model_DRF_df          ,file="pPandemie_model_DRF_df.RData")


####
#  Tabelle mit AUC und PR erstellen
##
# AUC

auc_GBM_df         <- pROC::ci.auc(pPandemie_model_GBM_df         $truth,pPandemie_model_GBM_df         $p1)
auc_stacked_df     <- pROC::ci.auc(pPandemie_model_stacked_df      $truth,pPandemie_model_stacked_df      $p1)
auc_XGBoost_df     <- pROC::ci.auc(pPandemie_model_XGBoost_df      $truth,pPandemie_model_XGBoost_df      $p1)
auc_GLM_df         <- pROC::ci.auc(pPandemie_model_GLM_df         $truth,pPandemie_model_GLM_df         $p1)
auc_DeepLearning_df<- pROC::ci.auc(pPandemie_model_DeepLearning_df$truth,pPandemie_model_DeepLearning_df$p1)
auc_DRF_df         <- pROC::ci.auc(pPandemie_model_DRF_df         $truth,pPandemie_model_DRF_df         $p1)


## PR
pr_GBM_df         <- PRROC::pr.curve(scores.class0 = pPandemie_model_GBM_df$p1[pPandemie_model_GBM_df$truth==1],  pPandemie_model_GBM_df$p1[pPandemie_model_GBM_df$truth==0], curve = TRUE)
pr_stacked_df     <- PRROC::pr.curve(scores.class0 = pPandemie_model_stacked_df$p1[pPandemie_model_stacked_df$truth==1], scores.class1 = pPandemie_model_stacked_df$p1[pPandemie_model_stacked_df$truth==0], curve = TRUE)
pr_XGBoost_df     <- PRROC::pr.curve(scores.class0 = pPandemie_model_XGBoost_df$p1[pPandemie_model_XGBoost_df$truth==1], scores.class1 = pPandemie_model_XGBoost_df$p1[pPandemie_model_XGBoost_df$truth==0], curve = TRUE)
pr_GLM_df         <- PRROC::pr.curve(scores.class0 = pPandemie_model_GLM_df$p1[pPandemie_model_GLM_df$truth==1]        , scores.class1 = pPandemie_model_GLM_df$p1[pPandemie_model_GLM_df$truth==0], curve = TRUE)
pr_DeepLearning_df<- PRROC::pr.curve(scores.class0 = pPandemie_model_DeepLearning_df$p1[pPandemie_model_DeepLearning_df$truth==1], scores.class1 = pPandemie_model_DeepLearning_df$p1[pPandemie_model_DeepLearning_df$truth==0], curve = TRUE)
pr_DRF_df         <- PRROC::pr.curve(scores.class0 = pPandemie_model_DRF_df$p1[pPandemie_model_DRF_df$truth==1], scores.class1 = pPandemie_model_DRF_df$p1[pPandemie_model_DRF_df$truth==0], curve = TRUE)






tabelle_ergebnisse <- data.frame(names=c("GBM"  ,       
                                         "stacked",     
                                         "XGBoost" ,    
                                         "GLM"     ,    
                                         "DeepLearning",
                                         "DRF"         ),
                                 AUC=c(paste(sprintf("%.3f",auc_GBM_df         [2])," [",sprintf("%.3f",auc_GBM_df         [1])," to ",sprintf("%.3f",auc_GBM_df         [3]),"]",sep=""),
                                       paste(sprintf("%.3f",auc_stacked_df     [2])," [",sprintf("%.3f",auc_stacked_df     [1])," to ",sprintf("%.3f",auc_stacked_df     [3]),"]",sep=""),
                                       paste(sprintf("%.3f",auc_XGBoost_df     [2])," [",sprintf("%.3f",auc_XGBoost_df     [1])," to ",sprintf("%.3f",auc_XGBoost_df     [3]),"]",sep=""),
                                       paste(sprintf("%.3f",auc_GLM_df         [2])," [",sprintf("%.3f",auc_GLM_df         [1])," to ",sprintf("%.3f",auc_GLM_df         [3]),"]",sep=""),
                                       paste(sprintf("%.3f",auc_DeepLearning_df[2])," [",sprintf("%.3f",auc_DeepLearning_df[1])," to ",sprintf("%.3f",auc_DeepLearning_df[3]),"]",sep=""),
                                       paste(sprintf("%.3f",auc_DRF_df         [2])," [",sprintf("%.3f",auc_DRF_df         [1])," to ",sprintf("%.3f",auc_DRF_df         [3]),"]",sep="")),
                                 PR=c(pr_ci(pr_GBM_df         ,length(pPandemie_model_GBM_df         $p1)),
                                      pr_ci(pr_stacked_df     ,length(pPandemie_model_stacked_df     $p1)),
                                      pr_ci(pr_XGBoost_df     ,length(pPandemie_model_XGBoost_df     $p1)),
                                      pr_ci(pr_GLM_df         ,length(pPandemie_model_GLM_df         $p1)),
                                      pr_ci(pr_DeepLearning_df,length(pPandemie_model_DeepLearning_df$p1)),
                                      pr_ci(pr_DRF_df         ,length(pPandemie_model_DRF_df         $p1))))

tabelle_ergebnisse
openxlsx::write.xlsx(tabelle_ergebnisse,file="tabelle_ergebnisse.xlsx")



###
# AUROC mit CI berechnen
###
pPandemie_df$truth <- pandemie$verstorben
pROC::roc(pPandemie_df$truth,pPandemie_df$p1)


aaa <- pROC::ci.auc(pPandemie_df$truth,pPandemie_df$p1)

paste(sprintf("%.3f",aaa[2]),
      " [",
      sprintf("%.3f",aaa[1]),
      " to ",
      sprintf("%.3f",aaa[3]),
      "]",sep="")

###
# AUPR
####

pr<- PRROC::pr.curve(scores.class0 = pPandemie_df$p1[pPandemie_df$truth==1],  pPandemie_df$p1[pPandemie_df$truth==0], curve = TRUE)


pr_ci(pr                                   ,length(pPandemie_df$p1))


perf_stacked <- h2o.performance(model_stacked, pandemie_h20)
perf_stacked




#####
#  Grafiken
#######
#  ROC


library(pROC)
library(ggsci)
library(ggplot2)


roc_val <- pROC::roc(pPandemie_df$truth,pPandemie_df$p1,ret="all_coords")
roc_val



roc_val_df<-data.frame(sensitivity=roc_val$sensitivities,
                       specificity=roc_val$specificities,
                       prediction=c(unique(pPandemie_df$p1),1))



ggplot(data=roc_val_df,aes(x=1-specificity,y=sensitivity))+geom_step(size=2)+
  theme_bw()+
  scale_y_continuous("Sensitivity",lim=c(0,1))+
  scale_x_continuous("1-Specificity") +
  theme(axis.title = element_text(size=20),
        axis.text = element_text(size=18,color="black"),
        panel.grid = element_blank())+
  annotate("text",x=0.65,y=0.25,label=paste("AUROC: ",sprintf("%.3f",pROC::ci.auc(pPandemie_df$truth,pPandemie_df$p1)[2]),
                                            " [",
                                            sprintf("%.3f",pROC::ci.auc(pPandemie_df$truth,pPandemie_df$p1)[1]),
                                            " to ",
                                            sprintf("%.3f",pROC::ci.auc(pPandemie_df$truth,pPandemie_df$p1)[3]),
                                            "]",sep=""), size = 8)


library(export)
graph2ppt(file=".../PowerPointGrafiken/ROC_GBM.pptx")
ggsave(".../ROCC_GBM.jpg")



#####
#  PR
####
prPandemie<- PRROC::pr.curve(scores.class0 = pPandemie_df$p1[pPandemie_df$truth==1],  pPandemie_df$p1[pPandemie_df$truth==0], curve = TRUE)

nu <- log(prPandemie$auc.integral/(1-prPandemie$auc.integral))
tau <- 1/sqrt(dim(prPandemie$curve)[1]*prPandemie$auc.integral*(1-prPandemie$auc.integral))

lower <- exp(nu-1.96*tau)/(1+exp(nu-1.96*tau))
lower

upper <- exp(nu+1.96*tau)/(1+exp(nu+1.96*tau))
upper



prPandemie_curve_df <- data.frame(prPandemie$curve)

names(prPandemie_curve_df) <- c("Recall","Precision","threshold")
head(prPandemie_curve_df)
summary(prPandemie_curve_df)
prPandemie_curve_df <- prPandemie_curve_df %>% arrange( Precision,Recall)
ggplot(data=prPandemie_curve_df %>% arrange( desc(Precision),Recall),aes(x=Recall,y=Precision))+
  geom_line(size=2)+
  theme_bw()+
  geom_hline(yintercept = 0.009773434  ,linetype=2)+
  scale_y_continuous(lim=c(0,1))+
  #scale_x_continuous() +
  theme(axis.title = element_text(size=20),
        axis.text = element_text(size=18,color="black"),
        panel.grid = element_blank())+
  annotate("text",x=0.6,y=0.6,label=paste("AUPR: ",sprintf("%.3f",prPandemie$auc.integral)," [",sprintf("%.3f",lower)," to ",round(upper,3),"]",sep=""), size = 8)



graph2ppt(file=".../PowerPointGrafiken/PR_GBM.pptx")
ggsave(".../PR_GBM.jpg")




#######
#  Baum
####


library(mlr)


baum <- ctree(factor(verstorben)~.,data=train_vorPandemie)
plot(baum)

####
#  Partial dependence plots
###



#####
#  PDP
##


train_vor_pandemie_model_GBM            <- h2o.predict(model_GBM         , newdata = train_vorPandemie_h20)
train_vor_pandemie_model_stacked        <- h2o.predict(model_stacked     , newdata = train_vorPandemie_h20)
train_vor_pandemie_model_XGBoost        <- h2o.predict(model_XGBoost     , newdata = train_vorPandemie_h20)
train_vor_pandemie_model_GLM            <- h2o.predict(model_GLM         , newdata = train_vorPandemie_h20)
train_vor_pandemie_model_DeepLearning   <- h2o.predict(model_DeepLearning, newdata = train_vorPandemie_h20)
train_vor_pandemie_model_DRF            <- h2o.predict(model_DRF         , newdata = train_vorPandemie_h20)

train_vor_pandemie_model_GBM_df         <- as.data.frame(train_vor_pandemie_model_GBM         )
train_vor_pandemie_model_stacked_df     <- as.data.frame(train_vor_pandemie_model_stacked     )
train_vor_pandemie_model_XGBoost_df     <- as.data.frame(train_vor_pandemie_model_XGBoost     )
train_vor_pandemie_model_GLM_df         <- as.data.frame(train_vor_pandemie_model_GLM         )
train_vor_pandemie_model_DeepLearning_df<- as.data.frame(train_vor_pandemie_model_DeepLearning)
train_vor_pandemie_model_DRF_df         <- as.data.frame(train_vor_pandemie_model_DRF         )

train_vor_pandemie_model_GBM_df         $truth <- train_vorPandemie$verstorben
train_vor_pandemie_model_stacked_df     $truth <- train_vorPandemie$verstorben
train_vor_pandemie_model_XGBoost_df     $truth <- train_vorPandemie$verstorben
train_vor_pandemie_model_GLM_df         $truth <- train_vorPandemie$verstorben
train_vor_pandemie_model_DeepLearning_df$truth <- train_vorPandemie$verstorben
train_vor_pandemie_model_DRF_df         $truth <- train_vorPandemie$verstorben


save(train_vor_pandemie_model_GBM_df          ,file="train_vor_pandemie_model_GBM_df.RData")    
save(train_vor_pandemie_model_stacked_df      ,file="train_vor_pandemie_model_stacked_df.RData")   
save(train_vor_pandemie_model_XGBoost_df      ,file="train_vor_pandemie_model_XGBoost_df.RData")  
save(train_vor_pandemie_model_GLM_df          ,file="train_vor_pandemie_model_GLM_df.RData") 
save(train_vor_pandemie_model_DeepLearning_df ,file="train_vor_pandemie_model_DeepLearning_df.RData")
save(train_vor_pandemie_model_DRF_df          ,file="train_vor_pandemie_model_DRF_df.RData")



#####
#  Test
##


test_vor_pandemie_model_GBM            <- h2o.predict(model_GBM         , newdata = test_vorPandemie_h20)
test_vor_pandemie_model_stacked        <- h2o.predict(model_stacked     , newdata = test_vorPandemie_h20)
test_vor_pandemie_model_XGBoost        <- h2o.predict(model_XGBoost     , newdata = test_vorPandemie_h20)
test_vor_pandemie_model_GLM            <- h2o.predict(model_GLM         , newdata = test_vorPandemie_h20)
test_vor_pandemie_model_DeepLearning   <- h2o.predict(model_DeepLearning, newdata = test_vorPandemie_h20)
test_vor_pandemie_model_DRF            <- h2o.predict(model_DRF         , newdata = test_vorPandemie_h20)

test_vor_pandemie_model_GBM_df         <- as.data.frame(test_vor_pandemie_model_GBM         )
test_vor_pandemie_model_stacked_df     <- as.data.frame(test_vor_pandemie_model_stacked     )
test_vor_pandemie_model_XGBoost_df     <- as.data.frame(test_vor_pandemie_model_XGBoost     )
test_vor_pandemie_model_GLM_df         <- as.data.frame(test_vor_pandemie_model_GLM         )
test_vor_pandemie_model_DeepLearning_df<- as.data.frame(test_vor_pandemie_model_DeepLearning)
test_vor_pandemie_model_DRF_df         <- as.data.frame(test_vor_pandemie_model_DRF         )

test_vor_pandemie_model_GBM_df         $truth <- test_vorPandemie$verstorben
test_vor_pandemie_model_stacked_df     $truth <- test_vorPandemie$verstorben
test_vor_pandemie_model_XGBoost_df     $truth <- test_vorPandemie$verstorben
test_vor_pandemie_model_GLM_df         $truth <- test_vorPandemie$verstorben
test_vor_pandemie_model_DeepLearning_df$truth <- test_vorPandemie$verstorben
test_vor_pandemie_model_DRF_df         $truth <- test_vorPandemie$verstorben


save(test_vor_pandemie_model_GBM_df          ,file="test_vor_pandemie_model_GBM_df.RData")    
save(test_vor_pandemie_model_stacked_df      ,file="test_vor_pandemie_model_stacked_df.RData")   
save(test_vor_pandemie_model_XGBoost_df      ,file="test_vor_pandemie_model_XGBoost_df.RData")  
save(test_vor_pandemie_model_GLM_df          ,file="test_vor_pandemie_model_GLM_df.RData") 
save(test_vor_pandemie_model_DeepLearning_df ,file="test_vor_pandemie_model_DeepLearning_df.RData")
save(test_vor_pandemie_model_DRF_df          ,file="test_vor_pandemie_model_DRF_df.RData")




predict_train_pandemie <- h2o.predict(model_stacked, newdata = train_vorPandemie_h20)
predict_train_df <- as.data.frame(predict_train)

l_df <- as.data.frame(l18)

wechsel_train_pdp <- wechsel_train %>% filter(is.na(wechsel_num)==FALSE) %>% select(wechsel_num,all_of(l18$Variable[1:10]),Labor_vorhanden)

predict_train_df <- cbind(predict_train_df,wechsel_train_pdp)
names(predict_train_df)


predict_train_df$Department <- NA
predict_train_df$Department[predict_train_df$Fachl.OE %in% c("ORS",	"UCS",	"UCM",	"ZOP")] <- "Bone&Joint"
predict_train_df$Department[predict_train_df$Fachl.OE %in% c("CHS"	,"GES",	"TCH",	"PCS")] <- "Surgery"
predict_train_df$Department[predict_train_df$Fachl.OE %in% c("HNS"	,"MKS",	"AUS"	)] <- "Head&Neck"
predict_train_df$Department[predict_train_df$Fachl.OE %in% c("NCS"	)		] <- "Neurosurgery"
predict_train_df$Department[predict_train_df$Fachl.OE %in% c("ZAC"	,"DES",	"PCM",	"SP")] <- "Outpatient"
predict_train_df$Department[predict_train_df$Fachl.OE %in% c("URS"	)		] <- "Urology"
predict_train_df$Department[predict_train_df$Fachl.OE %in% c("GY1")			] <- "Gyn/Obsetric"


library(ggplot2)

#1
ggplot(data=predict_train_df %>% filter(is.na(X5.550.1)==FALSE),aes(x=factor(X5.550.1),y=p1))+geom_boxplot()+theme_bw()+scale_y_continuous(trans="logit",n.breaks=10,"Prediction")+scale_x_discrete("Nephrostomie",labels=c("no","yes"))+theme(axis.title = element_text(size=18),axis.text = element_text(size=16))
graph2ppt(file=".../PowerPointGrafiken/pdp_1.pptx")
#2
ggplot(data=predict_train_df %>% filter(is.na(BMI_berechnet)==FALSE),aes(x=BMI_berechnet,y=p1))+geom_point()+geom_smooth()+theme_bw()+scale_y_continuous(trans="logit",n.breaks=10,"Prediction")+scale_x_continuous(n.breaks=10)+theme(axis.title = element_text(size=18),axis.text = element_text(size=16))
graph2ppt(file=".../PowerPointGrafiken/pdp_2.pptx")
#3
ggplot(data=predict_train_df,aes(x=OP_AGE,y=p1))+geom_point()+geom_smooth()+theme_bw()+scale_y_continuous(trans="logit",n.breaks=10,"Prediction")+scale_x_continuous(n.breaks=10)+theme(axis.title = element_text(size=18),axis.text = element_text(size=16))
graph2ppt(file=".../PowerPointGrafiken/pdp_3.pptx")

#4
ggplot(data=predict_train_df %>% filter(!Fachl.OE=="") %>% filter(is.na(Fachl.OE)==FALSE),aes(x=factor(Fachl.OE),y=p1))+geom_boxplot()+theme_bw()+scale_y_continuous(trans="logit",n.breaks=10,"Prediction")+theme(axis.title = element_text(size=18),axis.text = element_text(size=16))
graph2ppt(file=".../PowerPointGrafiken/pdp_4_fach.pptx")


ggplot(data=predict_train_df %>% filter(!Department=="") %>% filter(is.na(Department)==FALSE),aes(x=factor(Department),y=p1))+geom_boxplot()+theme_bw()+scale_y_continuous(trans="logit",n.breaks=10,"Prediction")+theme(axis.title = element_text(size=18),axis.text = element_text(size=16))
graph2ppt(file=".../PowerPointGrafiken/pdp_4_Department.pptx")

#5
ggplot(data=predict_train_df %>% filter(SEX != "U")  ,aes(x=factor(SEX),y=p1))+geom_boxplot()+theme_bw()+scale_y_continuous(trans="logit",n.breaks=10,"Prediction")+theme(axis.title = element_text(size=18),axis.text = element_text(size=16))
graph2ppt(file=".../PowerPointGrafiken/pdp_5.pptx")


#6
predict_train_df$Labor.Anforderung.EK_4p <- predict_train_df$Labor.Anforderung.EK
predict_train_df$Labor.Anforderung.EK_4p <- as.numeric(as.character(predict_train_df$Labor.Anforderung.EK_4p))
predict_train_df$Labor.Anforderung.EK_4p[predict_train_df$Labor.Anforderung.EK_4p>10] <- 10
predict_train_df$Labor.Anforderung.EK_4p[predict_train_df$Labor.Anforderung.EK_4p==5] <- 6
predict_train_df$Labor.Anforderung.EK_4p[predict_train_df$Labor.Anforderung.EK_4p==3] <- 4
predict_train_df$Labor.Anforderung.EK_4p[predict_train_df$Labor.Anforderung.EK_4p==1] <- 2




predict_train_df$Labor.Anforderung.EK_4p <-ifelse(is.na(predict_train_df$Labor.Anforderung.EK)==TRUE & predict_train_df$Labor_vorhanden==1,0,predict_train_df$Labor.Anforderung.EK_4p)
predict_train_df$Labor.Anforderung.EK_4p <- factor(predict_train_df$Labor.Anforderung.EK_4p)


ggplot(data=predict_train_df ,aes(x=factor(Labor.Anforderung.EK_4p),y=p1))+geom_boxplot()+theme_bw()+scale_y_continuous(trans="logit",n.breaks=10,"Prediction")+theme(axis.title = element_text(size=18),axis.text = element_text(size=16))+scale_x_discrete("EK orders")
graph2ppt(file=".../PowerPointGrafiken/pdp_6.pptx")

#7
predict_train_df$ASA_ein <- NA
predict_train_df$ASA_ein[predict_train_df$ASA==1] <- "1"
predict_train_df$ASA_ein[predict_train_df$ASA==2] <- "2"
predict_train_df$ASA_ein[predict_train_df$ASA==3] <- "3"
predict_train_df$ASA_ein[predict_train_df$ASA==4] <- "≥4"
predict_train_df$ASA_ein[predict_train_df$ASA==5] <- "≥4"
predict_train_df$ASA_ein[predict_train_df$ASA==9] <- "≥4"
predict_train_df$ASA_ein <- factor(predict_train_df$ASA_ein,levels=c("1","2","3","≥4"))

ggplot(data=predict_train_df,aes(x=factor(ASA_ein),y=p1))+geom_boxplot()+theme_bw()+scale_y_continuous(trans="logit",n.breaks=10)+theme(axis.title = element_text(size=18),axis.text = element_text(size=16))
graph2ppt(file=".../PowerPointGrafiken/pdp_7.pptx")

#8
predict_train_df$X5_ein <- ifelse(predict_train_df$X5>=10, "≥10",predict_train_df$X5)
predict_train_df$X5_ein <- factor(predict_train_df$X5_ein,levels=c("0","1","2","3","4","5","6","7","8","9","≥10"))
ggplot(data=predict_train_df,aes(x=factor(X5_ein),y=p1))+geom_boxplot()+theme_bw()+scale_y_continuous(trans="logit",n.breaks=10)+scale_x_discrete("Number of parallel Procedure codes")+theme(axis.title = element_text(size=18),axis.text = element_text(size=16))
graph2ppt(file=".../PowerPointGrafiken/pdp_8.pptx")

#9
ggplot(data=predict_train_df,aes(x=Labor.GFR..MDRD.,y=p1))+geom_point()+geom_smooth()+theme_bw()+scale_y_continuous(trans="logit",n.breaks=10)+scale_x_continuous(n.breaks=10)+theme(axis.title = element_text(size=18),axis.text = element_text(size=16))
graph2ppt(file=".../PowerPointGrafiken/pdp_9.pptx")


#10
ggplot(data=predict_train_df,aes(x=Labor.Leukozyten,y=p1))+geom_point()+geom_smooth()+theme_bw()+scale_y_continuous(trans="logit",n.breaks=10)+scale_x_continuous(n.breaks=10,lim=c(0,40))+theme(axis.title = element_text(size=18),axis.text = element_text(size=16))
graph2ppt(file=".../PowerPointGrafiken/pdp_10.pptx")



###
#. Variance bias
#

p_t_vor_pandemie  <- h2o.predict(StackedEnsemble_BestOfFamily_6_AutoML_1_20221205_81006, newdata = valid_vorPandemie_h20)
p_t_in_pandemie   <- h2o.predict(StackedEnsemble_BestOfFamily_6_AutoML_1_20221205_81006, newdata = pandemie_h20)




best_stacked      <- h2o.loadModel(paste("Mortality Covid AutoML/RunDauer 3Tage/Modellsicherung/Mortality_covid_",best_vor_Pandemie_Stacked$modelname,sep=""))
best_DeepLearning <- h2o.loadModel(paste("Mortality Covid AutoML/RunDauer 3Tage/Modellsicherung/Mortality_covid_",best_vor_Pandemie_DeepLearning$modelname,sep=""))
best_DR           <- h2o.loadModel(paste("Mortality Covid AutoML/RunDauer 3Tage/Modellsicherung/Mortality_covid_",best_vor_Pandemie_DRF         $modelname,sep=""))
best_GBM          <- h2o.loadModel(paste("Mortality Covid AutoML/RunDauer 3Tage/Modellsicherung/Mortality_covid_",best_vor_Pandemie_GBM         $modelname,sep=""))
best_GLM          <- h2o.loadModel(paste("Mortality Covid AutoML/RunDauer 3Tage/Modellsicherung/Mortality_covid_",best_vor_Pandemie_GLM         $modelname,sep=""))
best_XGBoost      <- h2o.loadModel(paste("Mortality Covid AutoML/RunDauer 3Tage/Modellsicherung/Mortality_covid_",best_vor_Pandemie_XGBoost     $modelname,sep=""))





h2o.predict(best_DeepLearning, newdata = train_vorPandemie_h20)

p_t_train_best_stacked             <- h2o.predict(best_stacked     , newdata = train_vorPandemie_h20)
p_t_train_best_DeepLearning        <- h2o.predict(best_DeepLearning, newdata = train_vorPandemie_h20)
p_t_train_best_DR                  <- h2o.predict(best_DR          , newdata = train_vorPandemie_h20)
p_t_train_best_GBM                 <- h2o.predict(best_GBM         , newdata = train_vorPandemie_h20)
p_t_train_best_GLM                 <- h2o.predict(best_GLM         , newdata = train_vorPandemie_h20)
p_t_train_best_XGBoost             <- h2o.predict(best_XGBoost     , newdata = train_vorPandemie_h20)

p_t_train_best_stacked_df             <- as.data.frame(p_t_train_best_stacked       )
p_t_train_best_DeepLearning_df        <- as.data.frame(p_t_train_best_DeepLearning  )
p_t_train_best_DR_df                  <- as.data.frame(p_t_train_best_DR            )
p_t_train_best_GBM_df                 <- as.data.frame(p_t_train_best_GBM           )
p_t_train_best_GLM_df                 <- as.data.frame(p_t_train_best_GLM           )
p_t_train_best_XGBoost_df             <- as.data.frame(p_t_train_best_XGBoost       )


###
#
###


predictions_vor_pandemie <- data.frame(truth=vorPandemie_validierung$verstorben)
predictions_in_pandemie <- data.frame(truth=pandemie$verstorben)


train_vorPandemie_predictions <- train_vorPandemie %>% select(verstorben)

train_vorPandemie_predictions$pred_stacked        <- as.numeric(p_t_train_best_stacked_df        $p1)
train_vorPandemie_predictions$pred_DeepLearning   <- as.numeric(p_t_train_best_DeepLearning_df   $p1)
train_vorPandemie_predictions$pred_DR             <- as.numeric(p_t_train_best_DR_df             $p1)
train_vorPandemie_predictions$pred_GBM            <- as.numeric(p_t_train_best_GBM_df            $p1)
train_vorPandemie_predictions$pred_GLM            <- as.numeric(p_t_train_best_GLM_df            $p1)
train_vorPandemie_predictions$pred_XGBoost        <- as.numeric(p_t_train_best_XGBoost_df        $p1)


####
#   Variance Bias trade of
###

vb_df <- data.frame(bias=rbind(
  
  bias_stacked     =mean(train_vorPandemie_predictions$pred_stacked     -train_vorPandemie_predictions$verstorben),
  bias_DeepLearning=mean(train_vorPandemie_predictions$pred_DeepLearning-train_vorPandemie_predictions$verstorben),
  bias_DRF          =mean(train_vorPandemie_predictions$pred_DR          -train_vorPandemie_predictions$verstorben),
  bias_GBM         =mean(train_vorPandemie_predictions$pred_GBM         -train_vorPandemie_predictions$verstorben),
  bias_GLM         =mean(train_vorPandemie_predictions$pred_GLM         -train_vorPandemie_predictions$verstorben),
  bias_XGBoost     =mean(train_vorPandemie_predictions$pred_XGBoost     -train_vorPandemie_predictions$verstorben)),
  
  variance=rbind(
    variance_stacked     =mean((train_vorPandemie_predictions$pred_stacked     -train_vorPandemie_predictions$verstorben)^2),
    variance_DeepLearning=mean((train_vorPandemie_predictions$pred_DeepLearning-train_vorPandemie_predictions$verstorben)^2),
    variance_DRF          =mean((train_vorPandemie_predictions$pred_DR          -train_vorPandemie_predictions$verstorben)^2),
    variance_GBM         =mean((train_vorPandemie_predictions$pred_GBM         -train_vorPandemie_predictions$verstorben)^2),
    variance_GLM         =mean((train_vorPandemie_predictions$pred_GLM         -train_vorPandemie_predictions$verstorben)^2),
    variance_XGBoost     =mean((train_vorPandemie_predictions$pred_XGBoost     -train_vorPandemie_predictions$verstorben)^2)),
  
  
  AUC_change=rbind(
    AUC_change_stacked     = Leaderboard_Pandemie$AUC_in_Pandemie[Leaderboard_Pandemie$modelname==best_vor_Pandemie_Stacked$modelname]-Leaderboard_Pandemie$AUC_vor_Pandemie[Leaderboard_Pandemie$modelname==best_vor_Pandemie_Stacked$modelname],
    AUC_change_DeepLearning= Leaderboard_Pandemie$AUC_in_Pandemie[Leaderboard_Pandemie$modelname==best_vor_Pandemie_DeepLearning$modelname]-Leaderboard_Pandemie$AUC_vor_Pandemie[Leaderboard_Pandemie$modelname==best_vor_Pandemie_DeepLearning$modelname],
    AUC_change_DRF          = Leaderboard_Pandemie$AUC_in_Pandemie[Leaderboard_Pandemie$modelname==best_vor_Pandemie_DRF         $modelname]-Leaderboard_Pandemie$AUC_vor_Pandemie[Leaderboard_Pandemie$modelname==best_vor_Pandemie_DRF         $modelname],
    AUC_change_GBM         = Leaderboard_Pandemie$AUC_in_Pandemie[Leaderboard_Pandemie$modelname==best_vor_Pandemie_GBM         $modelname]-Leaderboard_Pandemie$AUC_vor_Pandemie[Leaderboard_Pandemie$modelname==best_vor_Pandemie_GBM         $modelname],
    AUC_change_GLM         = Leaderboard_Pandemie$AUC_in_Pandemie[Leaderboard_Pandemie$modelname==best_vor_Pandemie_GLM         $modelname]-Leaderboard_Pandemie$AUC_vor_Pandemie[Leaderboard_Pandemie$modelname==best_vor_Pandemie_GLM         $modelname],
    AUC_change_XGBoost     = Leaderboard_Pandemie$AUC_in_Pandemie[Leaderboard_Pandemie$modelname==best_vor_Pandemie_XGBoost  $modelname]-Leaderboard_Pandemie$AUC_vor_Pandemie[Leaderboard_Pandemie$modelname==best_vor_Pandemie_XGBoost  $modelname]),
  
  PR_change=rbind(
    PR_change_stacked     = Leaderboard_Pandemie$PR_in_Pandemie[Leaderboard_Pandemie$modelname==best_vor_Pandemie_Stacked$modelname]-Leaderboard_Pandemie$PR_vor_Pandemie[Leaderboard_Pandemie$modelname==best_vor_Pandemie_Stacked$modelname],
    PR_change_DeepLearning= Leaderboard_Pandemie$PR_in_Pandemie[Leaderboard_Pandemie$modelname==best_vor_Pandemie_DeepLearning$modelname]-Leaderboard_Pandemie$PR_vor_Pandemie[Leaderboard_Pandemie$modelname==best_vor_Pandemie_DeepLearning$modelname],
    PR_change_DRF          = Leaderboard_Pandemie$PR_in_Pandemie[Leaderboard_Pandemie$modelname==best_vor_Pandemie_DRF         $modelname]-Leaderboard_Pandemie$PR_vor_Pandemie[Leaderboard_Pandemie$modelname==best_vor_Pandemie_DRF         $modelname],
    PR_change_GBM         = Leaderboard_Pandemie$PR_in_Pandemie[Leaderboard_Pandemie$modelname==best_vor_Pandemie_GBM         $modelname]-Leaderboard_Pandemie$PR_vor_Pandemie[Leaderboard_Pandemie$modelname==best_vor_Pandemie_GBM         $modelname],
    PR_change_GLM         = Leaderboard_Pandemie$PR_in_Pandemie[Leaderboard_Pandemie$modelname==best_vor_Pandemie_GLM         $modelname]-Leaderboard_Pandemie$PR_vor_Pandemie[Leaderboard_Pandemie$modelname==best_vor_Pandemie_GLM         $modelname],
    PR_change_XGBoost     = Leaderboard_Pandemie$PR_in_Pandemie[Leaderboard_Pandemie$modelname==best_vor_Pandemie_XGBoost  $modelname]-Leaderboard_Pandemie$PR_vor_Pandemie[Leaderboard_Pandemie$modelname==best_vor_Pandemie_XGBoost  $modelname]),
  
  Model=c("stacked"     ,
          "DeepLearning",
          "DRF"          ,
          "GBM"         ,
          "GLM"         ,
          "XGBoost"     )
  
  
)


vb_df <- data.frame(modelname=NA ,
                    bias=NA   ,
                    variance=NA,
                    AUC_diff=NA,
                    PR_diff=NA )


for(i in 47: dim(Leaderboard_Pandemie)[1])
{
  
  load_temp          <- h2o.loadModel(paste("Mortality Covid AutoML/RunDauer 3Tage/Modellsicherung/Mortality_covid_",Leaderboard_Pandemie$modelname[i],sep=""))
  predict_temp       <- h2o.predict(load_temp     , newdata = train_vorPandemie_h20)
  df_temp            <- as.data.frame(predict_temp       )
  bias_temp          <- mean(df_temp$p1-train_vorPandemie$verstorben)
  variance_temp      <- mean((df_temp$p1-train_vorPandemie$verstorben)^2)
  AUC_diff_temp      <-  Leaderboard_Pandemie$AUC_in_Pandemie[i]-Leaderboard_Pandemie$AUC_vor_Pandemie[i]
  PR_diff_temp      <-  Leaderboard_Pandemie$PR_in_Pandemie[i]-Leaderboard_Pandemie$PR_vor_Pandemie[i]
  
  vb_df <- rbind(vb_df,
                 c(Leaderboard_Pandemie$modelname[i],
                   bias_temp   ,   
                   variance_temp  ,
                   AUC_diff_temp  ,
                   PR_diff_temp   ))
  
  print(i)
  
}


vb_df_save <- vb_df


vb_df$bias  <- as.numeric(as.character(vb_df$bias ))
vb_df$variance <- as.numeric(as.character(vb_df$variance))
vb_df$AUC_diff <- as.numeric(as.character(vb_df$AUC_diff))
vb_df$PR_diff <- as.numeric(as.character(vb_df$PR_diff))




vb_df <- left_join(vb_df,Leaderboard_Pandemie %>% select(modelname,algo),"modelname")

save(vb_df,file="Bias variance AUC PR Change Covid Mortality.RData")

lb_vorPandemie_h20_df_Stacked      <- Leaderboard_Pandemie %>% filter(algo=="StackedEnsemble")
lb_vorPandemie_h20_df_DeepLearning <- lb_pandemie_df %>% filter(algo=="DeepLearning")
lb_vorPandemie_h20_df_DRF          <- lb_pandemie_df %>% filter(algo=="DRF")
lb_vorPandemie_h20_df_GBM          <- lb_pandemie_df %>% filter(algo=="GBM")
lb_vorPandemie_h20_df_GLM          <- lb_pandemie_df %>% filter(algo=="GLM")
lb_vorPandemie_h20_df_XGBoost      <- lb_pandemie_df %>% filter(algo=="XGBoost")

vb_df_best <- vb_df %>% filter(modelname %in% c(
  Leaderboard_Pandemie$modelname[Leaderboard_Pandemie$algo=="StackedEnsemble"][1],
  Leaderboard_Pandemie$modelname[Leaderboard_Pandemie$algo=="DeepLearning"][1],
  Leaderboard_Pandemie$modelname[Leaderboard_Pandemie$algo=="DeepLearning"][1],
  Leaderboard_Pandemie$modelname[Leaderboard_Pandemie$algo=="GBM"][1],
  Leaderboard_Pandemie$modelname[Leaderboard_Pandemie$algo=="GLM"][1],
  Leaderboard_Pandemie$modelname[Leaderboard_Pandemie$algo=="XGBoost"][1]
))

vb_df <- miceadds::load.data(file="Bias variance AUC PR Change Covid Mortality.RData",type="RData")

head(vb_df)

vb_df <- na.omit(vb_df)

library(ggplot2)

ggplot(data=vb_df,aes(x=AUC_diff,y=bias^2/variance,color=algo))+geom_point(size=4,shape=21,stroke=2,fill="white")+scale_y_continuous()+scale_x_continuous("Difference in AUC (pandemic time-pre pandemic time)")+theme_bw()+theme(axis.text = element_text(size=16,color="black"),axis.title=element_text(size=18),panel.grid = element_blank())

ggplot(data=vb_df,aes(x=AUC_diff,y=bias^2/variance,color=algo))+geom_point(size=4,shape=21,stroke=2,fill="white")+scale_y_continuous()+scale_x_continuous("Difference in AUC (pandemic time-pre pandemic time)")+theme_bw()+theme(axis.text = element_text(size=16,color="black"),axis.title=element_text(size=18),panel.grid = element_blank())+facet_wrap(~algo,scales="free")


ggplot(data=vb_df,aes(x=AUC_diff,y=bias^2/variance,color=algo))+geom_point(size=4,shape=21,stroke=2,fill="white")+scale_y_continuous(trans="logit")+scale_x_continuous("Difference in AUC (pandemic time-pre pandemic time)")+theme_classic()+theme(axis.text = element_text(size=16,color="black"),axis.title=element_text(size=18),panel.grid = element_blank())
ggplot(data=vb_df,aes(x=PR_diff,y=bias^2/variance,color=algo))+geom_point(size=4,shape=21,stroke=2,fill="white")+scale_y_continuous(trans="logit")+scale_x_continuous("Difference in AUPR (pandemic time-pre pandemic time)")+theme_classic()+theme(axis.text = element_text(size=16,color="black"),axis.title=element_text(size=18),panel.grid = element_blank())

ggplot(data=vb_df_best,aes(x=AUC_diff,y=bias^2/variance,color=algo))+geom_point(size=4,shape=21,stroke=2,fill="white")+scale_y_continuous(trans="logit")+scale_x_continuous("Difference in AUC (pandemic time-pre pandemic time)")+theme_classic()+theme(axis.text = element_text(size=16,color="black"),axis.title=element_text(size=18),panel.grid = element_blank())


ggplot(data=vb_df,aes(x=AUC_diff,y=PR_diff,color=algo,size=bias^2/variance))+geom_point(shape=21,stroke=2,fill="white")+
  scale_x_continuous("Difference in AUC (pandemic time-pre pandemic time)")+
  scale_y_continuous("Difference in AUPR (pandemic time-pre pandemic time)")+
  theme_classic()+theme(axis.text = element_text(size=16,color="black"),axis.title=element_text(size=18),panel.grid = element_blank())


library(export)

graph2ppt(file=".../Änderung AUC änderung PR BiasVariance.pptx")








###
# Veränderung der wichtigen Parameter
###




vi_vor_Pandemie_Stacked     $Variable[1:100]
vi_vor_Pandemie_DeepLearning$Variable[1:100]
vi_vor_Pandemie_DRF         $Variable[1:100]
vi_vor_Pandemie_GBM         $Variable[1:100]
vi_vor_Pandemie_GLM         $Variable[1:100]
vi_vor_Pandemie_XGBoost     $Variable[1:100]


top_10_vor_Stacked         <- vorPandemie %>% select(all_of(vi_vor_Pandemie_Stacked     $Variable[1:10]))
top_10_vor_DeepLearning    <- vorPandemie %>% select(all_of(vi_vor_Pandemie_DeepLearning$variable[1:10]))
top_10_vor_DRF             <- vorPandemie %>% select(all_of(vi_vor_Pandemie_DRF         $variable[1:10]))
top_10_vor_GBM             <- vorPandemie %>% select(all_of(vi_vor_Pandemie_GBM         $variable[1:10]))
top_10_vor_GLM             <- vorPandemie %>% select(all_of(vi_vor_Pandemie_GLM         $variable[1:10]))
top_10_vor_XGBoost         <- vorPandemie %>% select(all_of(vi_vor_Pandemie_XGBoost     $variable[1:10]))



top_10_pademie_Stacked     <- pandemie %>% select(all_of(vi_vor_Pandemie_Stacked     $Variable[1:10]))
top_10_pademie_DeepLearning<- pandemie %>% select(all_of(vi_vor_Pandemie_DeepLearning$variable[1:10]))
top_10_pademie_DRF         <- pandemie %>% select(all_of(vi_vor_Pandemie_DRF         $variable[1:10]))
top_10_pademie_GBM         <- pandemie %>% select(all_of(vi_vor_Pandemie_GBM         $variable[1:10]))
top_10_pademie_GLM         <- pandemie %>% select(all_of(vi_vor_Pandemie_GLM         $variable[1:10]))
top_10_pademie_XGBoost     <- pandemie %>% select(all_of(vi_vor_Pandemie_XGBoost     $variable[1:10]))


top_10_vor_Stacked         $time <- "vor"
top_10_vor_DeepLearning    $time <- "vor"
top_10_vor_DRF             $time <- "vor"
top_10_vor_GBM             $time <- "vor"
top_10_vor_GLM             $time <- "vor"
top_10_vor_XGBoost         $time <- "vor"
top_10_pademie_Stacked     $time <- "in"
top_10_pademie_DeepLearning$time <- "in"
top_10_pademie_DRF         $time <- "in"
top_10_pademie_GBM         $time <- "in"
top_10_pademie_GLM         $time <- "in"
top_10_pademie_XGBoost     $time <- "in"


top_10_Stacked      <- rbind(top_10_vor_Stacked     ,top_10_pademie_Stacked     )
top_10_DeepLearning <- rbind(top_10_vor_DeepLearning,top_10_pademie_DeepLearning)
top_10_DRF          <- rbind(top_10_vor_DRF         ,top_10_pademie_DRF         )
top_10_GBM          <- rbind(top_10_vor_GBM         ,top_10_pademie_GBM         )
top_10_GLM          <- rbind(top_10_vor_GLM         ,top_10_pademie_GLM         )
top_10_XGBoost      <- rbind(top_10_vor_XGBoost     ,top_10_pademie_XGBoost     )

save(top_10_Stacked      ,file="top_10_Stacked.RData")
save(top_10_DeepLearning ,file="top_10_DeepLearning.RData")
save(top_10_DRF          ,file="top_10_DRF.RData")
save(top_10_GBM          ,file="top_10_GBM.RData")
save(top_10_GLM          ,file="top_10_GLM.RData")
save(top_10_XGBoost      ,file="top_10_XGBoost.RData")

top_10_Stacked      <- miceadds::load.data(type="RData",file="top_10_Stacked.RData")
top_10_DeepLearning <- miceadds::load.data(type="RData",file="top_10_DeepLearning.RData")
top_10_DRF          <- miceadds::load.data(type="RData",file="top_10_DRF.RData")
top_10_GBM          <- miceadds::load.data(type="RData",file="top_10_GBM.RData")
top_10_GLM          <- miceadds::load.data(type="RData",file="top_10_GLM.RData")
top_10_XGBoost      <- miceadds::load.data(type="RData",file="top_10_XGBoost.RData")


names(top_10_Stacked     )
names(top_10_DeepLearning)
names(top_10_DRF         )
names(top_10_GBM         )
names(top_10_GLM         )
names(top_10_XGBoost     )


top_pandemie <- unique(c(names(top_10_Stacked     ),
                         names(top_10_DeepLearning),
                         names(top_10_DRF         ),
                         names(top_10_GBM         ),
                         names(top_10_GLM         ),
                         names(top_10_XGBoost     )))


library(dplyr)
top_10_vor_combiniert         <- vorPandemie %>% dplyr::select(verstorben,dplyr::any_of(top_pandemie),Operating_hours)
top_10_in_combiniert          <- pandemie    %>% dplyr::select(c(dplyr::any_of(top_pandemie),verstorben),Operating_hours)

top_10_vor_combiniert$time <- "vor"
top_10_in_combiniert$time  <- "in"


top_10_combiniert <- rbind(top_10_vor_combiniert,
                           top_10_in_combiniert )



top_10_combiniert$time <- factor(top_10_combiniert$time, levels = c("vor","in"))

library(gtsummary)

top_10_combiniert$Operating_hours <- 1-top_10_combiniert$Operating_hours
top_10_combiniert$Aufnahmegrund[top_10_combiniert$Aufnahmegrund==301] <- 101

top_10_combiniert$Aufnahmegrund <- case_when(top_10_combiniert$Aufnahmegrund==101~"vollstationär Normalfall",
                                             top_10_combiniert$Aufnahmegrund==102~"vollstationär Arbeitsunfall",
                                             top_10_combiniert$Aufnahmegrund==103~"vollstationär Unfall",
                                             top_10_combiniert$Aufnahmegrund==104~"vollstationär Hinweis auf Einwirkung von äußerer Gewalt",
                                             top_10_combiniert$Aufnahmegrund==107~"vollstationär Notfall",
                                             top_10_combiniert$Aufnahmegrund==102~"vollstationär Arbeitsunfall",
                                             top_10_combiniert$Aufnahmegrund==201~"vollstationär mit vorausgegangener vorstationärer Behandlung Normalfall",
                                             top_10_combiniert$Aufnahmegrund==202~"vollstationär mit vorausgegangener vorstationärer Behandlung Arbeitsunfall",
                                             top_10_combiniert$Aufnahmegrund==203~"vollstationär mit vorausgegangener vorstationärer Behandlung Unfall",
                                             top_10_combiniert$Aufnahmegrund==204~"vollstationär mit vorausgegangener vorstationärer Behandlung Hinweis auf Einwirkung von äußerer Gewalt",
                                             top_10_combiniert$Aufnahmegrund==207~"vollstationär mit vorausgegangener vorstationärer Behandlung Notfall",
                                             top_10_combiniert$Aufnahmegrund==202~"vollstationär mit vorausgegangener vorstationärer Behandlung Arbeitsunfall",
                                             top_10_combiniert$Aufnahmegrund==501~"Stationäre Entbindung Normalfall",
                                             top_10_combiniert$Aufnahmegrund==507~"Stationäre Entbindung Notfall",
                                             top_10_combiniert$Aufnahmegrund==801~"stationäre Aufnahme zur Organentnahme")



top_10_combiniert$hour <- factor(top_10_combiniert$hour)
top_10_combiniert$Labor.Anforderung.EK_ja    <- ifelse(top_10_combiniert$Labor.Anforderung.EK>0,"yes","no")
top_10_combiniert$Labor.Anforderung.EK_ifyes <-ifelse(top_10_combiniert$Labor.Anforderung.EK>0,top_10_DRF$Labor.Anforderung.EK,NA)
top_10_combiniert$Labor.Anforderung.GFP_ja    <- ifelse(top_10_combiniert$Labor.Anforderung.GFP>0,"yes","no")
top_10_combiniert$Labor.Anforderung.GFP_ifyes <-ifelse(top_10_combiniert$Labor.Anforderung.GFP>0,top_10_DRF$Labor.Anforderung.EK,NA)

top_10_combiniert$AnzahlKonislevorOP_keine <- ifelse(is.na(top_10_combiniert$AnzahlKonislevorOP)==TRUE,0,1)
top_10_combiniert$Labor.CRP..C.reakt..Protein._bestimmt <- ifelse(is.na(top_10_combiniert$Labor.CRP..C.reakt..Protein.)==TRUE,0,1) 
top_10_combiniert$Labor.QUICK.Wert_bestimmt <- ifelse(is.na(top_10_combiniert$Labor.QUICK.Wert)==TRUE,0,1)
top_10_combiniert$Labor.Hämoglobin_bestimmt <- ifelse(is.na(top_10_combiniert$Labor.Hämoglobin)==TRUE,0,1)

top_10_combiniert$Labor.Leukozyten_bestimmt <- ifelse(is.na(top_10_combiniert$Labor.Leukozyten)==TRUE,0,1)
top_10_combiniert$Labor.Thrombozyten_bestimmt <- ifelse(is.na(top_10_combiniert$Labor.Thrombozyten)==TRUE,0,1)
top_10_combiniert$Labor.Hämatokrit_bestimmt <- ifelse(is.na(top_10_combiniert$Labor.Hämatokrit)==TRUE,0,1)

top_10_combiniert$BMIorhanden <- ifelse(is.na(top_10_combiniert$BMI)==TRUE,0,1)
top_10_combiniert$ASAorhanden <- ifelse(is.na(top_10_combiniert$ASA)==TRUE,0,1)


top_10_combiniert$Department <- NA
top_10_combiniert$Department[top_10_combiniert$Fachl.OE %in% c(11,	18,	17,	21)] <- "Bone&Joint"
top_10_combiniert$Department[top_10_combiniert$Fachl.OE %in% c(3	,5,	16,	13)] <- "Surgery"
top_10_combiniert$Department[top_10_combiniert$Fachl.OE %in% c(7	,9,	2	)] <- "Head&Neck"
top_10_combiniert$Department[top_10_combiniert$Fachl.OE %in% c(10	)		] <- "Neurosurgery"
top_10_combiniert$Department[top_10_combiniert$Fachl.OE %in% c(20	,4,	12,	15,14)] <- "Outpatient"
top_10_combiniert$Department[top_10_combiniert$Fachl.OE %in% c(19	)		] <- "Urology"
top_10_combiniert$Department[top_10_combiniert$Fachl.OE %in% c(6)			] <- "Gyn/Obsetric"


top_10_combiniert %>% dplyr::select(time,verstorben,G,Alter,BMIorhanden,BMI,ASAorhanden,ASA,Aufnahmegrund,AnzahlKonislevorOP_keine,AnzahlKonislevorOP,Behandlungstage.vorstationär,Department,ORT_PREMED_vorhanden,
                                    Allergie_keine_bekannt_vorhanden,Zahnstatus_kariöses_Gebiß,
                                    Labor.Leukozyten_bestimmt,Labor.Leukozyten,Labor.QUICK.Wert_bestimmt,Labor.QUICK.Wert,Labor.Hämoglobin_bestimmt,Labor.Hämoglobin,
                                    Labor.CRP..C.reakt..Protein._bestimmt,Labor.CRP..C.reakt..Protein.,Labor.Hämatokrit_bestimmt,Labor.Hämatokrit,Labor.Thrombozyten_bestimmt,Labor.Thrombozyten,
                                    Labor.Anforderung.EK_ja,Labor.Anforderung.EK_ifyes,Labor.Anforderung.GFP_ja,Labor.Anforderung.GFP_ifyes,
                                    hour,Operating_hours,
                                    X5.984,X5.010.10,X5.022.00) %>% 
  tbl_summary(by=time,type=list(Aufnahmegrund ~ "categorical"),digits=list(all_categorical()~c(0,1)),
              label = list(X5.984~"Microsurgical technique",
                           X5.010.10~"Craniectomy (Calotte)",
                           X5.022.00~"Incision on the cerebrospinal fluid system (ventricular)")) %>% 
  add_p(test = list(all_categorical() ~ "fisher.test.simulate.p.values"))



###
# Tabelle mit Validierung vor Pandemie
###
library(dplyr)
top_10_vor_combiniert_v         <- vorPandemie_validierung %>% dplyr::select(verstorben,dplyr::any_of(top_pandemie),Operating_hours)
top_10_in_combiniert          <- pandemie    %>% dplyr::select(c(dplyr::any_of(top_pandemie),verstorben),Operating_hours)

top_10_vor_combiniert_v$time <- "vor"
top_10_in_combiniert$time  <- "in"


top_10_combiniert_v <- rbind(top_10_vor_combiniert_v,
                             top_10_in_combiniert )



top_10_combiniert_v$time <- factor(top_10_combiniert_v$time, levels = c("vor","in"))

library(gtsummary)

top_10_combiniert_v$Operating_hours <- 1-top_10_combiniert_v$Operating_hours
top_10_combiniert_v$Aufnahmegrund[top_10_combiniert_v$Aufnahmegrund==301] <- 101

top_10_combiniert_v$Aufnahmegrund <- case_when(top_10_combiniert_v$Aufnahmegrund==101~"vollstationär Normalfall",
                                               top_10_combiniert_v$Aufnahmegrund==102~"vollstationär Arbeitsunfall",
                                               top_10_combiniert_v$Aufnahmegrund==103~"vollstationär Unfall",
                                               top_10_combiniert_v$Aufnahmegrund==104~"vollstationär Hinweis auf Einwirkung von äußerer Gewalt",
                                               top_10_combiniert_v$Aufnahmegrund==107~"vollstationär Notfall",
                                               top_10_combiniert_v$Aufnahmegrund==102~"vollstationär Arbeitsunfall",
                                               top_10_combiniert_v$Aufnahmegrund==201~"vollstationär mit vorausgegangener vorstationärer Behandlung Normalfall",
                                               top_10_combiniert_v$Aufnahmegrund==202~"vollstationär mit vorausgegangener vorstationärer Behandlung Arbeitsunfall",
                                               top_10_combiniert_v$Aufnahmegrund==203~"vollstationär mit vorausgegangener vorstationärer Behandlung Unfall",
                                               top_10_combiniert_v$Aufnahmegrund==204~"vollstationär mit vorausgegangener vorstationärer Behandlung Hinweis auf Einwirkung von äußerer Gewalt",
                                               top_10_combiniert_v$Aufnahmegrund==207~"vollstationär mit vorausgegangener vorstationärer Behandlung Notfall",
                                               top_10_combiniert_v$Aufnahmegrund==202~"vollstationär mit vorausgegangener vorstationärer Behandlung Arbeitsunfall",
                                               top_10_combiniert_v$Aufnahmegrund==501~"Stationäre Entbindung Normalfall",
                                               top_10_combiniert_v$Aufnahmegrund==507~"Stationäre Entbindung Notfall",
                                               top_10_combiniert_v$Aufnahmegrund==801~"stationäre Aufnahme zur Organentnahme")



top_10_combiniert_v$hour <- factor(top_10_combiniert_v$hour)
top_10_combiniert_v$Labor.Anforderung.EK_ja    <- ifelse(top_10_combiniert_v$Labor.Anforderung.EK>0,"yes","no")
top_10_combiniert_v$Labor.Anforderung.EK_ifyes <-ifelse(top_10_combiniert_v$Labor.Anforderung.EK>0,top_10_DRF$Labor.Anforderung.EK,NA)
top_10_combiniert_v$Labor.Anforderung.GFP_ja    <- ifelse(top_10_combiniert_v$Labor.Anforderung.GFP>0,"yes","no")
top_10_combiniert_v$Labor.Anforderung.GFP_ifyes <-ifelse(top_10_combiniert_v$Labor.Anforderung.GFP>0,top_10_DRF$Labor.Anforderung.EK,NA)

top_10_combiniert_v$AnzahlKonislevorOP_keine <- ifelse(is.na(top_10_combiniert_v$AnzahlKonislevorOP)==TRUE,0,1)
top_10_combiniert_v$Labor.CRP..C.reakt..Protein._bestimmt <- ifelse(is.na(top_10_combiniert_v$Labor.CRP..C.reakt..Protein.)==TRUE,0,1) 
top_10_combiniert_v$Labor.QUICK.Wert_bestimmt <- ifelse(is.na(top_10_combiniert_v$Labor.QUICK.Wert)==TRUE,0,1)
top_10_combiniert_v$Labor.Hämoglobin_bestimmt <- ifelse(is.na(top_10_combiniert_v$Labor.Hämoglobin)==TRUE,0,1)

top_10_combiniert_v$Labor.Leukozyten_bestimmt <- ifelse(is.na(top_10_combiniert_v$Labor.Leukozyten)==TRUE,0,1)
top_10_combiniert_v$Labor.Thrombozyten_bestimmt <- ifelse(is.na(top_10_combiniert_v$Labor.Thrombozyten)==TRUE,0,1)
top_10_combiniert_v$Labor.Hämatokrit_bestimmt <- ifelse(is.na(top_10_combiniert_v$Labor.Hämatokrit)==TRUE,0,1)

top_10_combiniert_v$BMI_vorhanden <- ifelse(is.na(top_10_combiniert_v$BMI)==TRUE,0,1)
top_10_combiniert_v$ASA_vorhanden <- ifelse(is.na(top_10_combiniert_v$ASA)==TRUE,0,1)


top_10_combiniert_v$Department <- NA
top_10_combiniert_v$Department[top_10_combiniert_v$Fachl.OE %in% c(11,	18,	17,	21)] <- "Bone&Joint"
top_10_combiniert_v$Department[top_10_combiniert_v$Fachl.OE %in% c(3	,5,	16,	13)] <- "Surgery"
top_10_combiniert_v$Department[top_10_combiniert_v$Fachl.OE %in% c(7	,9,	2	)] <- "Head&Neck"
top_10_combiniert_v$Department[top_10_combiniert_v$Fachl.OE %in% c(10	)		] <- "Neurosurgery"
top_10_combiniert_v$Department[top_10_combiniert_v$Fachl.OE %in% c(20	,4,	12,	15,14)] <- "Outpatient"
top_10_combiniert_v$Department[top_10_combiniert_v$Fachl.OE %in% c(19	)		] <- "Urology"
top_10_combiniert_v$Department[top_10_combiniert_v$Fachl.OE %in% c(6)			] <- "Gyn/Obsetric"


top_10_combiniert_v %>% dplyr::select(time,verstorben,G,Alter,BMI_vorhanden,BMI,ASA_vorhanden,ASA,Aufnahmegrund,AnzahlKonislevorOP_keine,AnzahlKonislevorOP,Behandlungstage.vorstationär,Department,ORT_PREMED_vorhanden,
                                      Allergie_keine_bekannt_vorhanden,Zahnstatus_kariöses_Gebiß,
                                      Labor.Leukozyten_bestimmt,Labor.Leukozyten,Labor.QUICK.Wert_bestimmt,Labor.QUICK.Wert,Labor.Hämoglobin_bestimmt,Labor.Hämoglobin,
                                      Labor.CRP..C.reakt..Protein._bestimmt,Labor.CRP..C.reakt..Protein.,Labor.Hämatokrit_bestimmt,Labor.Hämatokrit,Labor.Thrombozyten_bestimmt,Labor.Thrombozyten,
                                      Labor.Anforderung.EK_ja,Labor.Anforderung.EK_ifyes,Labor.Anforderung.GFP_ja,Labor.Anforderung.GFP_ifyes,
                                      hour,Operating_hours,
                                      X5.984,X5.010.10,X5.022.00) %>% 
  tbl_summary(by=time,type=list(Aufnahmegrund ~ "categorical"),digits=list(all_categorical()~c(0,1)),
              label = list(X5.984~"Microsurgical technique",
                           X5.010.10~"Craniectomy (Calotte)",
                           X5.022.00~"Incision on the cerebrospinal fluid system (ventricular)")) %>% 
  add_p(test = list(all_categorical() ~ "fisher.test.simulate.p.values"))




fisher.test.simulate.p.values <- function(data, variable, by, ...) {
  result <- list()
  test_results <- stats::fisher.test(data[[variable]], data[[by]], simulate.p.value = TRUE)
  result$p <- test_results$p.value
  result$test <- test_results$method
  result
} 


###
#.   Prediction shift
#


names(Leaderboard_Pandemie)[11:14] <- c("Reihenfolge.AUC.vor","Reihenfolge.PR.vor","Reihenfolge.AUC.in","Reihenfolge.PR.in" )

lb_Stacked      <- Leaderboard_Pandemie %>% filter(algo=="StackedEnsemble")
lb_DeepLearning <- Leaderboard_Pandemie %>% filter(algo=="DeepLearning")
lb_DRF          <- Leaderboard_Pandemie %>% filter(algo=="DRF")
lb_GBM          <- Leaderboard_Pandemie %>% filter(algo=="GBM")
lb_GLM          <- Leaderboard_Pandemie %>% filter(algo=="GLM")
lb_XGBoost      <- Leaderboard_Pandemie %>% filter(algo=="XGBoost")

best_vor_Pandemie_Stacked      <- lb_Stacked     %>% arrange(desc(PR_vor_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_vor_Pandemie_DeepLearning <- lb_DeepLearning%>% arrange(desc(PR_vor_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_vor_Pandemie_DRF          <- lb_DRF         %>% arrange(desc(PR_vor_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_vor_Pandemie_GBM          <- lb_GBM         %>% arrange(desc(PR_vor_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_vor_Pandemie_GLM          <- lb_GLM         %>% arrange(desc(PR_vor_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)
best_vor_Pandemie_XGBoost      <- lb_XGBoost     %>% arrange(desc(PR_vor_Pandemie)) %>% filter(row_number()==1 ) %>% select(modelname)



best_stacked      <- h2o.loadModel(paste("Mortality Covid AutoML/RunDauer 3Tage/Modellsicherung/Mortality_covid_",best_vor_Pandemie_Stacked$modelname,sep=""))
best_DeepLearning <- h2o.loadModel(paste("Mortality Covid AutoML/RunDauer 3Tage/Modellsicherung/Mortality_covid_",best_vor_Pandemie_DeepLearning$modelname,sep=""))
best_DR           <- h2o.loadModel(paste("Mortality Covid AutoML/RunDauer 3Tage/Modellsicherung/Mortality_covid_",best_vor_Pandemie_DRF         $modelname,sep=""))
best_GBM          <- h2o.loadModel(paste("Mortality Covid AutoML/RunDauer 3Tage/Modellsicherung/Mortality_covid_",best_vor_Pandemie_GBM         $modelname,sep=""))
best_GLM          <- h2o.loadModel(paste("Mortality Covid AutoML/RunDauer 3Tage/Modellsicherung/Mortality_covid_",best_vor_Pandemie_GLM         $modelname,sep=""))
best_XGBoost      <- h2o.loadModel(paste("Mortality Covid AutoML/RunDauer 3Tage/Modellsicherung/Mortality_covid_",best_vor_Pandemie_XGBoost     $modelname,sep=""))





h2o.predict(best_DeepLearning, newdata = train_vorPandemie_h20)

p_t_valid_vor_best_stacked             <- h2o.predict(best_stacked     , newdata = valid_vorPandemie_h20)
p_t_valid_vor_best_DeepLearning        <- h2o.predict(best_DeepLearning, newdata = valid_vorPandemie_h20)
p_t_valid_vor_best_DR                  <- h2o.predict(best_DR          , newdata = valid_vorPandemie_h20)
p_t_valid_vor_best_GBM                 <- h2o.predict(best_GBM         , newdata = valid_vorPandemie_h20)
p_t_valid_vor_best_GLM                 <- h2o.predict(best_GLM         , newdata = valid_vorPandemie_h20)
p_t_valid_vor_best_XGBoost             <- h2o.predict(best_XGBoost     , newdata = valid_vorPandemie_h20)

p_t_valid_vor_best_stacked_df             <- as.data.frame(p_t_valid_vor_best_stacked     )
p_t_valid_vor_best_DeepLearning_df        <- as.data.frame(p_t_valid_vor_best_DeepLearning)
p_t_valid_vor_best_DR_df                  <- as.data.frame(p_t_valid_vor_best_DR          )
p_t_valid_vor_best_GBM_df                 <- as.data.frame(p_t_valid_vor_best_GBM         )
p_t_valid_vor_best_GLM_df                 <- as.data.frame(p_t_valid_vor_best_GLM         )
p_t_valid_vor_best_XGBoost_df             <- as.data.frame(p_t_valid_vor_best_XGBoost     )


p_t_valid_in_best_stacked             <- h2o.predict(best_stacked     , newdata = pandemie_h20)
p_t_valid_in_best_DeepLearning        <- h2o.predict(best_DeepLearning, newdata = pandemie_h20)
p_t_valid_in_best_DR                  <- h2o.predict(best_DR          , newdata = pandemie_h20)
p_t_valid_in_best_GBM                 <- h2o.predict(best_GBM         , newdata = pandemie_h20)
p_t_valid_in_best_GLM                 <- h2o.predict(best_GLM         , newdata = pandemie_h20)
p_t_valid_in_best_XGBoost             <- h2o.predict(best_XGBoost     , newdata = pandemie_h20)

p_t_valid_in_best_stacked_df             <- as.data.frame(p_t_valid_in_best_stacked     )
p_t_valid_in_best_DeepLearning_df        <- as.data.frame(p_t_valid_in_best_DeepLearning)
p_t_valid_in_best_DR_df                  <- as.data.frame(p_t_valid_in_best_DR          )
p_t_valid_in_best_GBM_df                 <- as.data.frame(p_t_valid_in_best_GBM         )
p_t_valid_in_best_GLM_df                 <- as.data.frame(p_t_valid_in_best_GLM         )
p_t_valid_in_best_XGBoost_df             <- as.data.frame(p_t_valid_in_best_XGBoost     )



###
#
###
p_t_valid_vor_best_stacked_df     $time <- "vor"
p_t_valid_vor_best_DeepLearning_df$time <- "vor"
p_t_valid_vor_best_DR_df          $time <- "vor"
p_t_valid_vor_best_GBM_df         $time <- "vor"
p_t_valid_vor_best_GLM_df         $time <- "vor"
p_t_valid_vor_best_XGBoost_df     $time <- "vor"

p_t_valid_vor_best_stacked_df     $algo <- "stacked"
p_t_valid_vor_best_DeepLearning_df$algo <- "DeepLearning"
p_t_valid_vor_best_DR_df          $algo <- "DRF"
p_t_valid_vor_best_GBM_df         $algo <- "GBM"
p_t_valid_vor_best_GLM_df         $algo <- "GLM"
p_t_valid_vor_best_XGBoost_df     $algo <- "XGBoost"

p_t_valid_in_best_stacked_df      $time <- "in"  
p_t_valid_in_best_DeepLearning_df $time <- "in"  
p_t_valid_in_best_DR_df           $time <- "in"  
p_t_valid_in_best_GBM_df          $time <- "in"  
p_t_valid_in_best_GLM_df          $time <- "in"  
p_t_valid_in_best_XGBoost_df      $time <- "in"  

p_t_valid_in_best_stacked_df     $algo <- "stacked"
p_t_valid_in_best_DeepLearning_df$algo <- "DeepLearning"
p_t_valid_in_best_DR_df          $algo <- "DRF"
p_t_valid_in_best_GBM_df         $algo <- "GBM"
p_t_valid_in_best_GLM_df         $algo <- "GLM"
p_t_valid_in_best_XGBoost_df     $algo <- "XGBoost"


best_stacked_df     <- rbind(p_t_valid_vor_best_stacked_df     ,p_t_valid_in_best_stacked_df     )
best_DeepLearning_df<- rbind(p_t_valid_vor_best_DeepLearning_df,p_t_valid_in_best_DeepLearning_df)
best_DR_df          <- rbind(p_t_valid_vor_best_DR_df          ,p_t_valid_in_best_DR_df          )
best_GBM_df         <- rbind(p_t_valid_vor_best_GBM_df         ,p_t_valid_in_best_GBM_df         )
best_GLM_df         <- rbind(p_t_valid_vor_best_GLM_df         ,p_t_valid_in_best_GLM_df         )
best_XGBoost_df     <- rbind(p_t_valid_vor_best_XGBoost_df     ,p_t_valid_in_best_XGBoost_df     )



best_gegenüber <- rbind(best_stacked_df     ,
                        best_DeepLearning_df,
                        best_DR_df          ,
                        best_GBM_df         ,
                        best_GLM_df         ,
                        best_XGBoost_df     )



ggplot(data=best_gegenüber,aes(x=p1,fill=time))+geom_density()+facet_wrap(~algo,scales="free")



predictions_vor_pandemie <- data.frame(truth=vorPandemie_validierung$verstorben)
predictions_in_pandemie <- data.frame(truth=pandemie$verstorben)


train_vorPandemie_predictions <- train_vorPandemie %>% select(verstorben)

train_vorPandemie_predictions$pred_stacked        <- as.numeric(p_t_train_best_stacked_df        $p1)
train_vorPandemie_predictions$pred_DeepLearning   <- as.numeric(p_t_train_best_DeepLearning_df   $p1)
train_vorPandemie_predictions$pred_DR             <- as.numeric(p_t_train_best_DR_df             $p1)
train_vorPandemie_predictions$pred_GBM            <- as.numeric(p_t_train_best_GBM_df            $p1)
train_vorPandemie_predictions$pred_GLM            <- as.numeric(p_t_train_best_GLM_df            $p1)
train_vorPandemie_predictions$pred_XGBoost        <- as.numeric(p_t_train_best_XGBoost_df        $p1)




